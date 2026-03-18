from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "adult_income.csv"
TARGET_COLUMN = "income"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, na_values=["?"])


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def main() -> None:
    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=">50K")
    precision = precision_score(y_test, y_pred, pos_label=">50K")
    recall = recall_score(y_test, y_pred, pos_label=">50K")
    matrix = confusion_matrix(y_test, y_pred, labels=["<=50K", ">50K"])

    print(f"Dataset shape: {df.shape}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print("\nEvaluation Metrics")
    print("-" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("\nConfusion Matrix")
    print("-" * 60)
    print("Rows = actual, Columns = predicted")
    print(pd.DataFrame(matrix, index=["<=50K", ">50K"], columns=["<=50K", ">50K"]).to_string())


if __name__ == "__main__":
    main()
