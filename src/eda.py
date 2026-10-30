
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURE_DIR, TABLE_DIR, TARGET, ID_COLUMN
from .features import engineer_features

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------
# 1. Data profile
# ---------------------------------------------------------

def save_data_profile(df):
    """Create a data quality profile."""

    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(dropna=True),
    }).sort_values("missing_percent", ascending=False)

    profile.to_csv(TABLE_DIR / "data_profile.csv")

    return profile


# ---------------------------------------------------------
# 2. Target distribution
# ---------------------------------------------------------

def plot_target_distribution(df, target=TARGET):
    """Plot the distribution of the churn target."""

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    counts = df[target].value_counts(dropna=False).sort_index()

    ax = counts.plot(kind="bar", figsize=(7, 4))

    ax.set_title("Target Distribution: Churn")
    ax.set_xlabel("Churn class (verify coding in dataset documentation)")
    ax.set_ylabel("Number of customers")

    plt.tight_layout()

    path = FIGURE_DIR / "target_distribution.png"

    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    return path


# ---------------------------------------------------------
# 3. Missing values
# ---------------------------------------------------------

def plot_missing_values(df, top_n=20):
    """Plot columns with the highest percentage of missing values."""

    missing = (
        df.isna().mean() * 100
    ).sort_values(ascending=False)

    missing = missing[missing > 0].head(top_n).sort_values()

    if missing.empty:
        print("No missing values were found.")
        return None

    ax = missing.plot(kind="barh", figsize=(9, 6))

    ax.set_title(f"Columns with Most Missing Values (Top {top_n})")
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Column")

    plt.tight_layout()

    path = FIGURE_DIR / "missing_values.png"

    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    return path


# ---------------------------------------------------------
# 4. Numerical features by churn
# ---------------------------------------------------------

def plot_numeric_by_target(df, features, target=TARGET):
    """Compare numerical feature distributions across churn groups."""

    created = []

    for feature in features:

        if feature not in df.columns or target not in df.columns:
            continue

        data = df[[feature, target]].copy()

        data[feature] = pd.to_numeric(
            data[feature], errors="coerce"
        )

        data = data.dropna()

        if data.empty:
            continue

        plt.figure(figsize=(7, 4))

        sns.boxplot(
            data=data,
            x=target,
            y=feature
        )

        plt.title(f"{feature} by {target}")
        plt.tight_layout()

        path = FIGURE_DIR / f"{feature}_by_{target}.png"

        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.show()
        plt.close()

        created.append(path)

    return created


# ---------------------------------------------------------
# 5. Numerical distributions
# ---------------------------------------------------------

def plot_distributions(df, features, zoom=True):
    """
    Plot numerical distributions.

    When zoom=True, the chart displays the 1st to 99th
    percentile range without deleting original observations.
    """

    created = []

    for feature in features:

        if feature not in df.columns:
            continue

        values = pd.to_numeric(
            df[feature], errors="coerce"
        ).dropna()

        if values.empty:
            continue

        if zoom:
            lower = values.quantile(0.01)
            upper = values.quantile(0.99)

            visible = values[
                values.between(lower, upper)
            ]
        else:
            visible = values

        if visible.empty:
            continue

        plt.figure(figsize=(8, 4))

        sns.histplot(
            visible,
            bins=30,
            kde=True
        )

        title_suffix = (
            "(1st–99th percentile)"
            if zoom else "(full range)"
        )

        plt.title(f"Distribution of {feature} {title_suffix}")
        plt.xlabel(feature)
        plt.ylabel("Number of customers")

        plt.tight_layout()

        path = FIGURE_DIR / f"{feature}_distribution.png"

        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.show()
        plt.close()

        created.append(path)

    return created


# ---------------------------------------------------------
# 6. Correlation heatmap
# ---------------------------------------------------------

def plot_correlation_heatmap(
    df,
    max_features=20,
    target=TARGET
):
    """
    Plot correlations between numerical features.

    Selects features based on their absolute correlation
    with the target, when the target is numeric.
    """

    numeric = df.select_dtypes(include="number")

    numeric = numeric.loc[
        :, numeric.nunique(dropna=True) > 1
    ]

    if numeric.shape[1] < 2:
        print("Not enough numerical features for a heatmap.")
        return None

    if target in numeric.columns:

        correlations = (
            numeric.corr()[target]
            .drop(target)
            .abs()
            .dropna()
        )

        selected = correlations.nlargest(
            max_features
        ).index.tolist()

        selected.append(target)

        numeric = numeric[selected]

    else:
        numeric = numeric.iloc[:, :max_features]

    corr = numeric.corr()

    plt.figure(figsize=(13, 10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False
    )

    plt.title("Correlation Heatmap: Original and Engineered Features")

    plt.tight_layout()

    path = FIGURE_DIR / "correlation_heatmap.png"

    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    return path


# ---------------------------------------------------------
# 7. Churn rate by categorical feature
# ---------------------------------------------------------

def plot_churn_by_category(
    df,
    feature,
    target=TARGET,
    top_n=10
):
    """Compare churn rates across categorical groups."""

    if feature not in df.columns or target not in df.columns:
        print(f"Column not found: {feature}")
        return None

    data = df[[feature, target]].dropna().copy()

    if data.empty:
        print(f"No usable records for {feature}.")
        return None

    top_categories = (
        data[feature]
        .value_counts()
        .head(top_n)
        .index
    )

    data = data[
        data[feature].isin(top_categories)
    ]

    rates = (
        data.groupby(feature, observed=True)[target]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=rates.index.astype(str),
        y=rates.values
    )

    plt.title(f"Churn Rate by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Churn rate")

    plt.xticks(rotation=45, ha="right")

    plt.ylim(0, 1)

    plt.tight_layout()

    path = FIGURE_DIR / f"churn_by_{feature}.png"

    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    return path


# ---------------------------------------------------------
# 8. Scatter plot
# ---------------------------------------------------------

def plot_scatter(
    df,
    x_feature,
    y_feature,
    target=TARGET
):
    """Explore relationships between two numerical features."""

    required = [x_feature, y_feature, target]

    if not all(col in df.columns for col in required):
        print("One or more required columns are missing.")
        return None

    data = df[required].copy()

    data[x_feature] = pd.to_numeric(
        data[x_feature], errors="coerce"
    )

    data[y_feature] = pd.to_numeric(
        data[y_feature], errors="coerce"
    )

    data = data.dropna()

    if len(data) > 5000:
        data = data.sample(
            5000,
            random_state=42
        )

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=data,
        x=x_feature,
        y=y_feature,
        hue=target,
        alpha=0.6
    )

    plt.title(f"{y_feature} vs {x_feature}")

    plt.tight_layout()

    path = FIGURE_DIR / f"{y_feature}_vs_{x_feature}.png"

    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    return path


# ---------------------------------------------------------
# 9. Engineered feature analysis
# ---------------------------------------------------------

def plot_engineered_features(df, target=TARGET):
    """
    Analyse the engineered numerical and categorical features.

    Feature engineering is performed on a copy of the supplied
    dataframe. The original dataframe is not modified.
    """

    engineered = engineer_features(df)

    engineered_numeric = [
        "revenue_change_ratio",
        "customer_care_usage_ratio",
        "usage_change_ratio",
        "recent_vs_six_month_usage",
        "revenue_per_minute",
    ]

    engineered_categorical = [
        "equipment_age_group",
        "tenure_group",
    ]

    numeric_present = [
        feature for feature in engineered_numeric
        if feature in engineered.columns
    ]

    categorical_present = [
        feature for feature in engineered_categorical
        if feature in engineered.columns
    ]

    print("\nEngineered numerical features:")
    print(numeric_present)

    print("\nEngineered categorical features:")
    print(categorical_present)

    # Plot engineered numerical distributions.
    plot_distributions(
        engineered,
        numeric_present,
        zoom=True
    )

    # Compare engineered numerical features by churn.
    plot_numeric_by_target(
        engineered,
        numeric_present,
        target=target
    )

    # Compare engineered categorical groups by churn.
    for feature in categorical_present:
        plot_churn_by_category(
            engineered,
            feature,
            target=target
        )

    # Save engineered feature profile.
    profile_columns = (
        numeric_present + categorical_present
    )

    if profile_columns:
        profile = pd.DataFrame({
            "dtype": engineered[profile_columns].dtypes.astype(str),
            "missing_count": engineered[profile_columns].isna().sum(),
            "missing_percent": (
                engineered[profile_columns].isna().mean() * 100
            ).round(2),
            "unique_values": engineered[profile_columns].nunique(
                dropna=True
            ),
        })

        profile.to_csv(
            TABLE_DIR / "engineered_feature_profile.csv"
        )

        return engineered, profile

    return engineered, pd.DataFrame()


# ---------------------------------------------------------
# 10. Feature summary
# ---------------------------------------------------------

def numeric_summary(df):
    """Save descriptive statistics for numerical columns."""

    summary = df.select_dtypes(
        include="number"
    ).describe().T

    summary.to_csv(
        TABLE_DIR / "numeric_summary.csv"
    )

    return summary


def engineered_feature_summary(df, target=TARGET):
    """
    Save a summary of engineered features, including their
    relationship with the binary target.
    """

    engineered = engineer_features(df)

    numeric_features = [
        "revenue_change_ratio",
        "customer_care_usage_ratio",
        "usage_change_ratio",
        "recent_vs_six_month_usage",
        "revenue_per_minute",
    ]

    rows = []

    for feature in numeric_features:

        if feature not in engineered.columns:
            continue

        values = pd.to_numeric(
            engineered[feature], errors="coerce"
        )

        rows.append({
            "Feature": feature,
            "Mean": values.mean(),
            "Median": values.median(),
            "Missing_Percent": values.isna().mean() * 100,
            "Unique_Values": values.nunique(),
        })

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.round(3)

        summary.to_csv(
            TABLE_DIR / "engineered_feature_summary.csv",
            index=False
        )

    return summary
