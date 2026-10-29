import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from .config import TARGET, ID_COLUMN, RANDOM_STATE, TEST_SIZE, TABLE_DIR

def split_features_target(df, target=TARGET, id_column=ID_COLUMN):
    
    clean = df.dropna(subset=[target]).copy()
    y = clean[target]
    if y.nunique() != 2:
        raise ValueError(f"{target} must have exactly two classes.")

    unique = set(pd.Series(y).dropna().unique().tolist())
    if unique == {0, 1}:
        y = y.astype(int)
    else:
        classes = sorted(unique, key=lambda value: str(value))
        mapping = {classes[0]: 0, classes[1]: 1}
        print(f"Target mapping used: {mapping}")
        y = y.map(mapping).astype(int)

    drop_cols = [target]
    if id_column in clean.columns:
        drop_cols.append(id_column)
    X = clean.drop(columns=drop_cols)

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant_cols)
    return X, y, constant_cols

def build_preprocessor(X, scale_numeric=False):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols

def train_compare_models(df, target=TARGET):
    
    X, y, constant_cols = split_features_target(df, target=target)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor_lr, numeric_cols, categorical_cols = build_preprocessor(
        X_train, scale_numeric=True
    )
    preprocessor_rf, _, _ = build_preprocessor(X_train, scale_numeric=False)

    models = {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", preprocessor_lr),
            ("classifier", LogisticRegression(
                max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor_rf),
            ("classifier", RandomForestClassifier(
                n_estimators=180,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    fitted = {}
    predictions = {}
    rows = []
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        fitted[name] = pipeline
        predictions[name] = {"predicted": pred, "probability": prob}
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, prob),
            "PR_AUC_Average_Precision": average_precision_score(y_test, prob),
        })

    metrics = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
    metrics.to_csv(TABLE_DIR / "model_comparison.csv", index=False)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "models": fitted, "predictions": predictions,
        "metrics": metrics, "constant_columns_removed": constant_cols,
        "numeric_features": numeric_cols, "categorical_features": categorical_cols,
    }

def feature_importance(fitted_pipeline, top_n=20):

    classifier = fitted_pipeline.named_steps["classifier"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    if not hasattr(classifier, "feature_importances_"):
        raise ValueError("This estimator does not expose tree feature_importances_.")
    names = preprocessor.get_feature_names_out()
    values = classifier.feature_importances_
    result = pd.DataFrame({"Feature": names, "Importance": values})
    result = result.sort_values("Importance", ascending=False).head(top_n)
    result.to_csv(TABLE_DIR / "feature_importance.csv", index=False)
    return result
