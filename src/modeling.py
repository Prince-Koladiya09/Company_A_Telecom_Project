
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from .config import TARGET, ID_COLUMN, RANDOM_STATE, TEST_SIZE, TABLE_DIR
from .features import engineer_features


def split_features_target(df, target=TARGET, id_column=ID_COLUMN):
    """Engineer features, separate target, and remove customer identifiers."""

    if target not in df.columns:
        raise ValueError(f"Target column {target!r} was not found.")

    clean = df.dropna(subset=[target]).copy()
    y = clean[target]

    if y.nunique() != 2:
        raise ValueError("The target must contain exactly two classes.")

    unique = set(y.unique().tolist())

    if unique == {0, 1}:
        y = y.astype(int)
    else:
        classes = sorted(unique, key=lambda value: str(value))
        mapping = {classes[0]: 0, classes[1]: 1}
        print("Target mapping:", mapping)
        y = y.map(mapping).astype(int)

    # Create features before splitting. These transformations use only
    # information from the same customer's observation-period row.
    X = clean.drop(columns=[target])

    if id_column in X.columns:
        X = X.drop(columns=[id_column])

    X = engineer_features(X)

    # Exclude constant columns.
    constant_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=False) <= 1
    ]
    X = X.drop(columns=constant_cols)

    return X, y, constant_cols


def build_preprocessor(X, scale_numeric=False, hgb_mode=False):
    

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    if hgb_mode:
        # HistGradientBoosting can handle missing numerical values itself.
        # Categorical columns are imputed and ordinal-encoded because
        # this estimator does not directly accept raw text categories.
        numeric_pipe = "passthrough"

        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(
                strategy="constant",
                fill_value="missing"
            )),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )),
        ])

    else:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        numeric_pipe = Pipeline(steps=numeric_steps)

        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(
                strategy="constant",
                fill_value="missing"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20
            )),
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def choose_threshold(y_true, probabilities):
    
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def get_model_pipelines(X):
    """Return all six candidate model pipelines."""

    standard_preprocessor, _, _ = build_preprocessor(
        X, scale_numeric=True
    )
    tree_preprocessor, _, _ = build_preprocessor(
        X, scale_numeric=False
    )
    hgb_preprocessor, _, _ = build_preprocessor(
        X, hgb_mode=True
    )

    models = {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", standard_preprocessor),
            ("classifier", LogisticRegression(
                max_iter=2000,
                C=1.0,
                random_state=RANDOM_STATE,
            )),
        ]),

        "Decision Tree": Pipeline(steps=[
            ("preprocessor", tree_preprocessor),
            ("classifier", DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            )),
        ]),

        "Random Forest": Pipeline(steps=[
            ("preprocessor", tree_preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "AdaBoost": Pipeline(steps=[
            ("preprocessor", tree_preprocessor),
            ("classifier", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=1,
                    random_state=RANDOM_STATE,
                ),
                n_estimators=100,
                learning_rate=0.5,
                random_state=RANDOM_STATE,
            )),
        ]),

        "Gradient Boosting": Pipeline(steps=[
            ("preprocessor", tree_preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )),
        ]),

        "HistGradientBoosting": Pipeline(steps=[
            ("preprocessor", hgb_preprocessor),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    return models


def train_compare_models(df, target=TARGET):

    X, y, constant_cols = split_features_target(df, target=target)

    # Outer holdout: the test set is not used for model or threshold selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Inner validation split for decision-threshold selection.
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    candidate_models = get_model_pipelines(X_train)

    fitted_models = {}
    predictions = {}
    thresholds = {}
    rows = []

    for name, pipeline in candidate_models.items():
        print(f"\nTraining {name}...")

        # Fit on inner training data.
        pipeline.fit(X_fit, y_fit)

        # Choose threshold from validation probabilities.
        valid_prob = pipeline.predict_proba(X_valid)[:, 1]
        threshold = choose_threshold(y_valid, valid_prob)

        # Refit using the entire outer training set.
        pipeline.fit(X_train, y_train)

        # Evaluate probabilities on the untouched test set.
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)

        fitted_models[name] = pipeline
        thresholds[name] = threshold

        predictions[name] = {
            "predicted": test_pred,
            "probability": test_prob,
            "threshold": threshold,
        }

        rows.append({
            "Model": name,
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, test_pred),
            "Precision": precision_score(
                y_test, test_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, test_pred, zero_division=0
            ),
            "F1": f1_score(
                y_test, test_pred, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(y_test, test_prob),
            "PR_AUC_Average_Precision": average_precision_score(
                y_test, test_prob
            ),
        })

        print(
            f"{name}: validation threshold={threshold:.3f}, "
            f"test ROC-AUC={rows[-1]['ROC_AUC']:.4f}"
        )

    metrics = pd.DataFrame(rows).sort_values(
        "ROC_AUC", ascending=False
    ).reset_index(drop=True)

    metrics.to_csv(
        TABLE_DIR / "model_comparison.csv",
        index=False
    )

    metrics[["Model", "Threshold"]].to_csv(
        TABLE_DIR / "model_thresholds.csv",
        index=False
    )

    _, numeric_cols, categorical_cols = build_preprocessor(X_train)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": fitted_models,
        "predictions": predictions,
        "thresholds": thresholds,
        "metrics": metrics,
        "constant_columns_removed": constant_cols,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
    }


def feature_importance(fitted_pipeline, top_n=20):

    classifier = fitted_pipeline.named_steps["classifier"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]

    if not hasattr(classifier, "feature_importances_"):
        raise ValueError(
            "This estimator does not expose feature_importances_."
        )

    names = preprocessor.get_feature_names_out()
    values = classifier.feature_importances_

    result = pd.DataFrame({
        "Feature": names,
        "Importance": values,
    })

    result = result.sort_values(
        "Importance", ascending=False
    ).head(top_n)

    result.to_csv(
        TABLE_DIR / "feature_importance.csv",
        index=False
    )

    return result
