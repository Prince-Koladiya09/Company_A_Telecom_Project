import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURE_DIR, TABLE_DIR, TARGET, ID_COLUMN

sns.set_theme(style="whitegrid")

def save_data_profile(df):
    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(dropna=True),
    }).sort_values("missing_percent", ascending=False)
    profile.to_csv(TABLE_DIR / "data_profile.csv")
    return profile

def plot_target_distribution(df, target=TARGET):
    counts = df[target].value_counts(dropna=False).sort_index()
    ax = counts.plot(kind="bar", figsize=(7, 4))
    ax.set_title("Target distribution: churn")
    ax.set_xlabel("Churn (0 = no, 1 = yes; confirm coding in data)")
    ax.set_ylabel("Number of records")

    plt.tight_layout()
    path = FIGURE_DIR / "target_distribution.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()
    
    return path

def plot_missing_values(df, top_n=20):
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n).sort_values()

    if missing.empty:
        print("No missing values were found.")
        return None
    
    ax = missing.plot(kind="barh", figsize=(9, 6))
    ax.set_title(f"Columns with the most missing values (top {top_n})")
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Column")

    plt.tight_layout()
    path = FIGURE_DIR / "missing_values.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    return path

def plot_numeric_by_target(df, features, target=TARGET):
    created = []
    for feature in features:
        if feature not in df.columns:
            continue

        data = df[[feature, target]].copy()
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        data = data.dropna()

        if data.empty:
            continue

        plt.figure(figsize=(7, 4))
        sns.boxplot(data=data, x=target, y=feature)
        plt.title(f"{feature} by {target}")
        plt.tight_layout()

        path = FIGURE_DIR / f"{feature}_by_{target}.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.show()
        plt.close()

        created.append(path)
    return created

def plot_distributions(df, features):
    """Plot histograms for selected numerical features."""

    for feature in features:
        if feature not in df.columns:
            continue

        values = pd.to_numeric(df[feature], errors="coerce").dropna()

        if values.empty:
            continue

        plt.figure(figsize=(7, 4))
        sns.histplot(values, bins=30, kde=True)

        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Number of customers")
        plt.tight_layout()

        path = FIGURE_DIR / f"{feature}_distribution.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.show()
        plt.close()

def plot_correlation_heatmap(df, max_features=15):
    """Plot correlations between selected numerical features."""

    numeric = df.select_dtypes(include="number")

    # Remove constant columns
    numeric = numeric.loc[:, numeric.nunique() > 1]

    if numeric.shape[1] < 2:
        print("Not enough numerical features for a correlation heatmap.")
        return None

    # Select features with the highest absolute correlation with churn
    if TARGET in numeric.columns:
        correlations = numeric.corr()[TARGET].drop(TARGET).abs()
        selected = correlations.nlargest(max_features).index.tolist()
        selected.append(TARGET)
        numeric = numeric[selected]
    else:
        numeric = numeric.iloc[:, :max_features]

    corr = numeric.corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    path = FIGURE_DIR / "correlation_heatmap.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    return path

def plot_churn_by_category(df, feature, target=TARGET, top_n=10):
    """Compare churn rates across categories."""

    if feature not in df.columns or target not in df.columns:
        print(f"Column not found: {feature}")
        return None

    data = df[[feature, target]].dropna().copy()

    if data.empty:
        print("No usable records for this comparison.")
        return None

    # Keep the most frequent categories to avoid an unreadable chart
    top_categories = data[feature].value_counts().head(top_n).index
    data = data[data[feature].isin(top_categories)]

    rates = data.groupby(feature, observed=True)[target].mean()
    rates = rates.sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(x=rates.index.astype(str), y=rates.values)

    plt.title(f"Churn Rate by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Average target value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = FIGURE_DIR / f"churn_by_{feature}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    return path

def plot_scatter(df, x_feature, y_feature, target=TARGET):
    """Explore the relationship between two numerical features."""

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

    # Sample large datasets to keep the plot readable
    if len(data) > 5000:
        data = data.sample(5000, random_state=42)

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

def numeric_summary(df):
    summary = df.select_dtypes(include="number").describe().T
    summary.to_csv(TABLE_DIR / "numeric_summary.csv")
    return summary
