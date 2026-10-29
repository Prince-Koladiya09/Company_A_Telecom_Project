from src.data import load_data, check_target
from src.eda import (
    save_data_profile,
    plot_target_distribution,
    plot_missing_values,
    numeric_summary,
    plot_numeric_by_target,
    plot_distributions,
    plot_correlation_heatmap,
    plot_churn_by_category,
    plot_scatter,
)
from src.modeling import train_compare_models, feature_importance
from src.evaluation import evaluate_selected_model
from src.business import make_risk_ranking, top_group_lift
from src.narrative import metrics_summary, business_summary


def main():
    # 1. Load and merge data
    client, record, df = load_data()

    print("Client shape:", client.shape)
    print("Record shape:", record.shape)
    print("Merged shape:", df.shape)

    # 2. Check target distribution
    print("\nTarget counts and proportions:")
    check_target(df)

    # 3. Data quality and summary
    profile = save_data_profile(df)
    print("\nData profile:")
    print(profile.head(10))

    print("\nNumerical summary:")
    print(numeric_summary(df))

    # 4. Exploratory data analysis
    plot_target_distribution(df)
    plot_missing_values(df)

    eda_features = [
        "rev_Mean",
        "mou_Mean",
        "custcare_Mean",
        "change_mou",
        "change_rev",
        "eqpdays",
        "months",
    ]

    plot_numeric_by_target(df, eda_features)
    plot_distributions(df, eda_features)
    plot_correlation_heatmap(df)

    # Categorical comparisons: run only if the columns exist
    for feature in ["new_cell", "creditcd"]:
        if feature in df.columns:
            plot_churn_by_category(df, feature)

    # Scatter plot: run only if both columns exist
    if "mou_Mean" in df.columns and "rev_Mean" in df.columns:
        plot_scatter(df, "mou_Mean", "rev_Mean")

    # 5. Train and compare machine learning models
    model_results = train_compare_models(df)

    print("\nModel comparison:")
    print(model_results["metrics"].to_string(index=False))

    # 6. Evaluate the best-performing model
    best_name = model_results["metrics"].iloc[0]["Model"]

    evaluate_selected_model(model_results, best_name)

    print("\nModel summary draft:")
    print(metrics_summary(model_results["metrics"].iloc[0]))

    # 7. Feature importance
    if best_name == "Random Forest":
        importance = feature_importance(
            model_results["models"][best_name]
        )
        print("\nTop features:")
        print(importance.to_string(index=False))

    # 8. Business analysis: identify high-risk customers
    ranking = make_risk_ranking(model_results, best_name)
    lift = top_group_lift(ranking, share=0.10)

    print("\nTop-risk group summary:")
    print(lift)
    print(business_summary(lift))

    print("\nSaved tables and charts in the outputs/ folder.")


if __name__ == "__main__":
    main()