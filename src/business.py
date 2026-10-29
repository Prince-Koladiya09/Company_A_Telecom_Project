import numpy as np
import pandas as pd

from .config import TABLE_DIR

def make_risk_ranking(results, model_name, target_name="Actual_Churn"):
    """Rank held-out test records by predicted class-1 probability.

    The dataset has no customer names in this output. IDs are not used as model
    features; row indices are retained only to align predictions with source rows.
    """
    y_test = results["y_test"]
    y_prob = results["predictions"][model_name]["probability"]
    ranking = pd.DataFrame({
        target_name: np.asarray(y_test),
        "Predicted_Risk": np.asarray(y_prob),
    }, index=y_test.index).sort_values("Predicted_Risk", ascending=False)
    ranking["Risk_Decile"] = pd.qcut(
        ranking["Predicted_Risk"].rank(method="first"),
        q=10, labels=False
    ) + 1
    ranking.to_csv(TABLE_DIR / "test_set_risk_ranking.csv", index=True)
    return ranking

def top_group_lift(ranking, share=0.10):
    """Calculate descriptive capture in the highest-risk share of the test set."""
    if not 0 < share <= 1:
        raise ValueError("share must be in the interval (0, 1].")
    n = max(1, int(np.ceil(len(ranking) * share)))
    top = ranking.head(n)
    total_positive = ranking["Actual_Churn"].sum()
    capture = (top["Actual_Churn"].sum() / total_positive) if total_positive else np.nan
    result = {
        "test_rows": int(len(ranking)),
        "target_share": float(share),
        "selected_rows": int(n),
        "observed_positive_rate_top_group": float(top["Actual_Churn"].mean()),
        "observed_positive_rate_full_test": float(ranking["Actual_Churn"].mean()),
        "share_of_actual_positives_captured": float(capture),
    }
    return result

def scenario_calculator(customer_count, target_share, intervention_cost,
                        incremental_retention_rate, value_per_retained_customer):
    """Scenario calculator; inputs are assumptions, not company facts."""
    targeted = int(round(customer_count * target_share))
    retained = targeted * incremental_retention_rate
    cost = targeted * intervention_cost
    benefit = retained * value_per_retained_customer
    return {
        "customer_count": customer_count,
        "targeted_customers": targeted,
        "estimated_incremental_retained": retained,
        "campaign_cost": cost,
        "estimated_benefit": benefit,
        "estimated_net_benefit": benefit - cost,
        "assumptions": {
            "target_share": target_share,
            "intervention_cost": intervention_cost,
            "incremental_retention_rate": incremental_retention_rate,
            "value_per_retained_customer": value_per_retained_customer,
        },
    }
