import json

import joblib
from sklearn.metrics import classification_report, confusion_matrix

from config import MODEL_METADATA_PATH, SCALER_PATH, TEST_PATH, TRAIN_PATH
from preprocess import prepare_feature_data


def main():
    x_train, x_test, y_train, y_test, encoders = prepare_feature_data(
        TRAIN_PATH,
        TEST_PATH
    )

    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(metadata["best_selector_path"])
    model = joblib.load(metadata["best_model_path"])
    threshold = metadata["best_threshold"]

    x_test = scaler.transform(x_test)
    x_test = selector.transform(x_test)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print(f"Best Model: {metadata['best_model_name']}")
    print(f"Decision Threshold: {threshold:.2f}")
    print("Classification Report")
    print("=" * 40)
    print(classification_report(y_test, predictions, target_names=["Normal", "Attack"]))

    print("Confusion Matrix")
    print("=" * 40)
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
