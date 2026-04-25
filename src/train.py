import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from config import (
    COLUMNS,
    ENCODERS_PATH,
    FEATURE_SELECTOR_PATH,
    MODEL_DIR,
    MODEL_METADATA_PATH,
    NEURAL_NETWORK_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    REPORT_DIR,
    SCALER_PATH,
    TEST_PATH,
    TRAIN_PATH,
    TUNING_RESULTS_PATH,
    XGBOOST_MODEL_PATH,
    resolve_project_path,
)
from preprocess import prepare_feature_data, save_preprocessors, scale_features
from visualize import (
    plot_confusion,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


K_VALUES = [35, 38, 40]
THRESHOLDS = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
CV_SPLITS = 3


def build_selector(k):
    return Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
        ]
    )


def fit_selector(x_train, y_train, requested_k):
    variance = VarianceThreshold()
    x_train_var = variance.fit_transform(x_train)

    actual_k = min(requested_k, x_train_var.shape[1])
    select = SelectKBest(score_func=f_classif, k=actual_k)
    x_train_selected = select.fit_transform(x_train_var, y_train)

    selector = Pipeline(
        [
            ("variance", variance),
            ("select", select),
        ]
    )
    return selector, x_train_selected, actual_k


def score_predictions(y_true, predictions):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1 Score": f1_score(y_true, predictions, zero_division=0),
    }


def is_better_result(candidate, current_best):
    candidate_key = (
        candidate["F1 Score"],
        candidate["Recall"],
        candidate["Precision"],
        candidate["Accuracy"],
    )
    current_key = (
        current_best["F1 Score"],
        current_best["Recall"],
        current_best["Precision"],
        current_best["Accuracy"],
    )
    return candidate_key > current_key


def make_random_forest(params):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=1,
        **params,
    )


def make_neural_network(params):
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=200,
        early_stopping=True,
        random_state=42,
        **params,
    )


def make_xgboost(params):
    return XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        **params,
    )


def summarize_threshold_metrics(threshold_metrics):
    summary = {}
    for key in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        summary[key] = float(np.mean([metrics[key] for metrics in threshold_metrics]))
    return summary


def choose_best_cv_threshold(fold_probabilities, fold_targets):
    best = None

    for threshold in THRESHOLDS:
        threshold_metrics = []

        for probabilities, y_true in zip(fold_probabilities, fold_targets):
            predictions = (probabilities >= threshold).astype(int)
            threshold_metrics.append(score_predictions(y_true, predictions))

        candidate = {
            "threshold": threshold,
            **summarize_threshold_metrics(threshold_metrics),
        }

        if best is None or is_better_result(candidate, best):
            best = candidate

    return best


def cross_validate_model(model_name, builder, param_grid, x_train, y_train):
    records = []
    best_config = None
    splitter = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)

    for k in K_VALUES:
        for params in param_grid:
            fold_probabilities = []
            fold_targets = []

            for train_indices, val_indices in splitter.split(x_train, y_train):
                x_fold_train = x_train[train_indices]
                x_fold_val = x_train[val_indices]
                y_fold_train = y_train.iloc[train_indices]
                y_fold_val = y_train.iloc[val_indices]

                x_fold_train, x_fold_val, _ = scale_features(x_fold_train, x_fold_val)
                selector, x_fold_train, actual_k = fit_selector(
                    x_fold_train,
                    y_fold_train,
                    k,
                )
                x_fold_val = selector.transform(x_fold_val)

                model = builder(params)
                model.fit(x_fold_train, y_fold_train)

                fold_probabilities.append(model.predict_proba(x_fold_val)[:, 1])
                fold_targets.append(y_fold_val.to_numpy())

            threshold_result = choose_best_cv_threshold(fold_probabilities, fold_targets)

            record = {
                "Model": model_name,
                "k": actual_k,
                "params": json.dumps(params, sort_keys=True),
                "cv_splits": CV_SPLITS,
                **threshold_result,
            }
            records.append(record)

            candidate = {
                "model_name": model_name,
                "k": actual_k,
                "params": params,
                **threshold_result,
            }
            if best_config is None or is_better_result(candidate, best_config):
                best_config = candidate

    return best_config, records


def get_selected_feature_names(selector):
    raw_feature_names = np.array(COLUMNS[:-2])
    after_variance = raw_feature_names[selector.named_steps["variance"].get_support()]
    return after_variance[selector.named_steps["select"].get_support()]


def evaluate_model(name, model, x_test, y_test, threshold):
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = score_predictions(y_test, predictions)

    results = {
        "Model": name,
        "Threshold": threshold,
        **metrics,
    }

    print(f"\n{name} Results")
    print("=" * 40)
    print(f"Decision Threshold: {threshold:.2f}")
    print(classification_report(y_test, predictions, target_names=["Normal", "Attack"]))

    return results, predictions, probabilities


def selector_path_for(model_name):
    safe_name = model_name.lower().replace(" ", "_")
    return os.path.join(MODEL_DIR, f"{safe_name}_selector.pkl")


def main():
    os.makedirs(resolve_project_path(MODEL_DIR), exist_ok=True)
    os.makedirs(resolve_project_path(REPORT_DIR), exist_ok=True)

    x_train_raw, x_test_raw, y_train, y_test, encoders = prepare_feature_data(
        TRAIN_PATH,
        TEST_PATH
    )

    x_train_raw = x_train_raw.to_numpy(dtype=float)
    x_test_raw = x_test_raw.to_numpy(dtype=float)

    tuning_records = []

    rf_best, rf_records = cross_validate_model(
        "Random Forest",
        make_random_forest,
        [
            {"class_weight": "balanced"},
            {"class_weight": {0: 1.0, 1: 1.2}},
            {"class_weight": {0: 1.0, 1: 1.4}},
        ],
        x_train_raw,
        y_train,
    )
    tuning_records.extend(rf_records)

    nn_best, nn_records = cross_validate_model(
        "Neural Network",
        make_neural_network,
        [{}],
        x_train_raw,
        y_train,
    )
    tuning_records.extend(nn_records)

    xgb_best = None
    if XGBClassifier is not None:
        xgb_best, xgb_records = cross_validate_model(
            "XGBoost",
            make_xgboost,
            [
                {
                    "n_estimators": 280,
                    "max_depth": 5,
                    "learning_rate": 0.08,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "scale_pos_weight": 1.0,
                },
                {
                    "n_estimators": 300,
                    "max_depth": 6,
                    "learning_rate": 0.08,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "scale_pos_weight": 1.1,
                },
                {
                    "n_estimators": 320,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "scale_pos_weight": 1.1,
                },
                {
                    "n_estimators": 360,
                    "max_depth": 6,
                    "learning_rate": 0.06,
                    "subsample": 0.85,
                    "colsample_bytree": 0.8,
                    "scale_pos_weight": 1.2,
                },
            ],
            x_train_raw,
            y_train,
        )
        tuning_records.extend(xgb_records)

    tuning_results = pd.DataFrame(tuning_records).sort_values(
        by=["F1 Score", "Recall", "Precision", "Accuracy"],
        ascending=False,
    )
    tuning_results.to_csv(resolve_project_path(TUNING_RESULTS_PATH), index=False)
    print(f"Tuning summary saved to {TUNING_RESULTS_PATH}")

    best_cv_config = rf_best
    for candidate in [nn_best, xgb_best]:
        if candidate is not None and is_better_result(candidate, best_cv_config):
            best_cv_config = candidate

    x_train_scaled, x_test, scaler = scale_features(x_train_raw, x_test_raw)
    save_preprocessors(scaler, encoders)
    joblib.dump(encoders, resolve_project_path(ENCODERS_PATH))
    joblib.dump(scaler, resolve_project_path(SCALER_PATH))

    final_configs = [
        (rf_best, make_random_forest, RANDOM_FOREST_MODEL_PATH),
        (nn_best, make_neural_network, NEURAL_NETWORK_MODEL_PATH),
    ]
    if xgb_best is not None:
        final_configs.append((xgb_best, make_xgboost, XGBOOST_MODEL_PATH))

    final_results = []
    final_metadata = {}

    for config, builder, model_path in final_configs:
        selector, x_train_selected, actual_k = fit_selector(
            x_train_scaled,
            y_train,
            config["k"],
        )
        x_test_selected = selector.transform(x_test)
        selected_feature_names = get_selected_feature_names(selector)

        model = builder(config["params"])
        print(f"Training {config['model_name']} with tuned settings...")
        model.fit(x_train_selected, y_train)

        results, predictions, probabilities = evaluate_model(
            config["model_name"],
            model,
            x_test_selected,
            y_test,
            config["threshold"],
        )
        final_results.append(results)

        selector_path = selector_path_for(config["model_name"])
        joblib.dump(model, resolve_project_path(model_path))
        joblib.dump(selector, resolve_project_path(selector_path))

        final_metadata[config["model_name"]] = {
            "model_path": model_path,
            "selector_path": selector_path,
            "threshold": config["threshold"],
            "k": actual_k,
            "params": config["params"],
            "cv_metrics": {
                "Accuracy": config["Accuracy"],
                "Precision": config["Precision"],
                "Recall": config["Recall"],
                "F1 Score": config["F1 Score"],
            },
            "test_metrics": {
                "Accuracy": results["Accuracy"],
                "Precision": results["Precision"],
                "Recall": results["Recall"],
                "F1 Score": results["F1 Score"],
            },
            "selected_features": selected_feature_names.tolist(),
        }

        safe_name = config["model_name"].lower().replace(" ", "_")
        plot_confusion(
            y_test,
            predictions,
            f"{config['model_name']} Confusion Matrix",
            resolve_project_path(os.path.join(REPORT_DIR, f"{safe_name}_confusion_matrix.png")),
        )
        plot_roc(
            y_test,
            probabilities,
            f"{config['model_name']} ROC Curve",
            resolve_project_path(os.path.join(REPORT_DIR, f"{safe_name}_roc_curve.png")),
        )
        if hasattr(model, "feature_importances_"):
            plot_feature_importance(
                model,
                selected_feature_names,
                resolve_project_path(os.path.join(REPORT_DIR, f"{safe_name}_feature_importance.png")),
            )

    comparison = pd.DataFrame(final_results).sort_values(
        by=["F1 Score", "Recall", "Precision", "Accuracy"],
        ascending=False,
    )
    comparison.to_csv(
        resolve_project_path(os.path.join(REPORT_DIR, "model_comparison.csv")),
        index=False,
    )
    plot_model_comparison(
        comparison,
        resolve_project_path(os.path.join(REPORT_DIR, "model_comparison_bar_chart.png")),
    )

    production_model_info = final_metadata[best_cv_config["model_name"]]
    pd.Series(production_model_info["selected_features"], name="selected_feature").to_csv(
        resolve_project_path(os.path.join(REPORT_DIR, "selected_features.csv")),
        index=False,
    )
    joblib.dump(
        joblib.load(resolve_project_path(production_model_info["selector_path"])),
        resolve_project_path(FEATURE_SELECTOR_PATH),
    )

    metadata = {
        "best_model_name": best_cv_config["model_name"],
        "best_model_path": production_model_info["model_path"],
        "best_selector_path": production_model_info["selector_path"],
        "best_threshold": production_model_info["threshold"],
        "selection_basis": "cross_validated_f1",
        "models": final_metadata,
    }
    with open(resolve_project_path(MODEL_METADATA_PATH), "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print("\nBest production model selected from cross-validation:")
    print(
        f"{best_cv_config['model_name']} | CV F1={best_cv_config['F1 Score']:.4f} "
        f"| CV Recall={best_cv_config['Recall']:.4f} | Threshold={best_cv_config['threshold']:.2f}"
    )
    print("Model comparison saved to reports/model_comparison.csv")
    print("Tuning results saved to reports/tuning_results.csv")


if __name__ == "__main__":
    main()
