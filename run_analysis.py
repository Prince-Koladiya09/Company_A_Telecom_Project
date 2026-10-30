
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
    plot_engineered_features,
    engineered_feature_summary,
)

from src.modeling import train_compare_models, feature_importance
from src.evaluation import evaluate_selected_model
from src.business import make_risk_ranking, top_group_lift
from src.narrative import metrics_summary, business_summary


def main():

    # -----------------------------------------------------
    # 1. Load and validate data
    # -----------------------------------------------------

    client, record, df = load_data()

    print("\n========== DATA OVERVIEW ==========")
    print("Client shape:", client.shape)
    print("Record shape:", record.shape)
    print("Merged shape:", df.shape)

    print("\nTarget counts and proportions:")

    target_counts, target_proportions = check_target(df)

    print("\nTarget counts:")
    print(target_counts)

    print("\nTarget proportions:")
    print(target_proportions)

    # -----------------------------------------------------
    # 2. Data profile and original EDA
    # -----------------------------------------------------

    print("\n========== DATA PROFILE ==========")

    print("\nSaving data profile...")

    profile = save_data_profile(df)

    print(profile.head(10))

    print("\nNumerical summary:")
    print(numeric_summary(df))

    # Target distribution
    print("\n========== ORIGINAL EDA ==========")

    plot_target_distribution(df)
    plot_missing_values(df)

    # Select available original numerical features
    eda_features = [
        "rev_Mean",
        "mou_Mean",
        "custcare_Mean",
        "change_mou",
        "change_rev",
        "eqpdays",
        "months",
    ]

    eda_features = [
        feature
        for feature in eda_features
        if feature in df.columns
    ]

    if eda_features:
        plot_numeric_by_target(df, eda_features)
        plot_distributions(df, eda_features)

    # Correlation analysis on original features
    plot_correlation_heatmap(df)

    # Categorical feature analysis
    for feature in ["new_cell", "creditcd"]:
        if feature in df.columns:
            plot_churn_by_category(df, feature)

    # Scatter plot
    if "mou_Mean" in df.columns and "rev_Mean" in df.columns:
        plot_scatter(df, "mou_Mean", "rev_Mean")

    # -----------------------------------------------------
    # 3. EDA for engineered features
    # -----------------------------------------------------

    print("\n========== ENGINEERED FEATURE EDA ==========")

    # Generate engineered features and their visualisations
    engineered_df, engineered_profile = plot_engineered_features(df)

    print("\nEngineered feature profile:")
    print(engineered_profile.head(15))

    # Generate engineered feature summary
    engineered_summary = engineered_feature_summary(df)

    print("\nEngineered feature summary:")
    print(engineered_summary.to_string(index=False))

    # Correlation analysis including engineered features
    plot_correlation_heatmap(engineered_df)

    # -----------------------------------------------------
    # 4. Train and compare six models
    # -----------------------------------------------------

    print("\n========== MODEL TRAINING ==========")

    model_results = train_compare_models(df)

    print("\nModel comparison:")
    print(model_results["metrics"].to_string(index=False))

    # Select the best model based on the sorted metrics
    best_name = model_results["metrics"].iloc[0]["Model"]

    # -----------------------------------------------------
    # 5. Evaluate selected model
    # -----------------------------------------------------

    print("\n========== MODEL EVALUATION ==========")

    evaluate_selected_model(model_results, best_name)

    print("\nModel summary:")
    print(metrics_summary(model_results["metrics"].iloc[0]))

    # -----------------------------------------------------
    # 6. Feature importance, when supported
    # -----------------------------------------------------

    print("\n========== FEATURE IMPORTANCE ==========")

    try:
        importance = feature_importance(
            model_results["models"][best_name]
        )

        print("\nTop features:")
        print(importance.to_string(index=False))

    except (ValueError, AttributeError, KeyError) as error:
        print("\nFeature importance unavailable:", error)

    # -----------------------------------------------------
    # 7. Business risk ranking
    # -----------------------------------------------------

    print("\n========== BUSINESS ANALYSIS ==========")

    ranking = make_risk_ranking(model_results, best_name)

    lift = top_group_lift(
        ranking,
        share=0.10,
    )

    print("\nTop-risk group summary:")
    print(lift)

    print("\nBusiness interpretation:")
    print(business_summary(lift))

    # -----------------------------------------------------
    # 8. Selected model and decision threshold
    # -----------------------------------------------------

    print("\n========== FINAL MODEL SELECTION ==========")

    print("\nSelected model:", best_name)

    print(
        "Selected threshold:",
        model_results["thresholds"][best_name],
    )

    # -----------------------------------------------------
    # 9. Completion
    # -----------------------------------------------------

    print("\n========== ANALYSIS COMPLETED ==========")

    print("Saved tables in: outputs/tables/")
    print("Saved charts in: outputs/figures/")


if __name__ == "__main__":
    main()
