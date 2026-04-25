import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    CATEGORICAL_COLUMNS,
    COLUMNS,
    ENCODERS_PATH,
    SCALER_PATH,
    resolve_project_path,
)


def load_data(path):
    df = pd.read_csv(resolve_project_path(path), names=COLUMNS)
    df = df.drop("difficulty", axis=1)
    return df


def convert_to_binary_label(df):
    df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)
    return df


def encode_categorical(train_df, test_df):
    encoders = {}

    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()

        train_df[col] = encoder.fit_transform(train_df[col])

        known_classes = list(encoder.classes_)

        test_df[col] = test_df[col].apply(
            lambda x: x if x in known_classes else "unknown"
        )

        if "unknown" not in known_classes:
            encoder.classes_ = np.append(encoder.classes_, "unknown")

        test_df[col] = encoder.transform(test_df[col])
        encoders[col] = encoder

    return train_df, test_df, encoders


def prepare_data(train_path, test_path):
    X_train, X_test, y_train, y_test, encoders = prepare_feature_data(
        train_path,
        test_path
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders


def prepare_feature_data(train_path, test_path):
    train_df = load_data(train_path)
    test_df = load_data(test_path)

    train_df = convert_to_binary_label(train_df)
    test_df = convert_to_binary_label(test_df)

    train_df, test_df, encoders = encode_categorical(train_df, test_df)

    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    return X_train, X_test, y_train, y_test, encoders


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def save_preprocessors(scaler, encoders):
    joblib.dump(scaler, resolve_project_path(SCALER_PATH))
    joblib.dump(encoders, resolve_project_path(ENCODERS_PATH))
