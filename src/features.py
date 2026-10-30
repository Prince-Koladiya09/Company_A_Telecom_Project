
import numpy as np
import pandas as pd


def safe_ratio(numerator, denominator):
    
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator.abs()
    return result.replace([np.inf, -np.inf], np.nan)


def engineer_features(df):

    data = df.copy()

    # 1. Revenue change relative to historical revenue
    if {"change_rev", "avg3rev"}.issubset(data.columns):
        data["revenue_change_ratio"] = safe_ratio(
            pd.to_numeric(data["change_rev"], errors="coerce"),
            pd.to_numeric(data["avg3rev"], errors="coerce"),
        )

    # 2. Customer-care usage relative to total usage
    if {"custcare_Mean", "mou_Mean"}.issubset(data.columns):
        data["customer_care_usage_ratio"] = safe_ratio(
            pd.to_numeric(data["custcare_Mean"], errors="coerce"),
            pd.to_numeric(data["mou_Mean"], errors="coerce"),
        )

    # 3. Change in usage relative to historical usage
    if {"change_mou", "avg3mou"}.issubset(data.columns):
        data["usage_change_ratio"] = safe_ratio(
            pd.to_numeric(data["change_mou"], errors="coerce"),
            pd.to_numeric(data["avg3mou"], errors="coerce"),
        )

    # 4. Recent usage relative to six-month usage
    if {"avg3mou", "avg6mou"}.issubset(data.columns):
        data["recent_vs_six_month_usage"] = safe_ratio(
            pd.to_numeric(data["avg3mou"], errors="coerce"),
            pd.to_numeric(data["avg6mou"], errors="coerce"),
        )

    # 5. Revenue per minute of usage
    if {"rev_Mean", "mou_Mean"}.issubset(data.columns):
        data["revenue_per_minute"] = safe_ratio(
            pd.to_numeric(data["rev_Mean"], errors="coerce"),
            pd.to_numeric(data["mou_Mean"], errors="coerce"),
        )

    # 6. Equipment age groups
    if "eqpdays" in data.columns:
        eqpdays = pd.to_numeric(data["eqpdays"], errors="coerce")
        data["equipment_age_group"] = pd.cut(
            eqpdays,
            bins=[-np.inf, 180, 365, 730, 1095, np.inf],
            labels=[
                "0-180 days",
                "181-365 days",
                "1-2 years",
                "2-3 years",
                "Over 3 years",
            ],
        ).astype("object")

    # 7. Customer tenure groups
    if "months" in data.columns:
        months = pd.to_numeric(data["months"], errors="coerce")
        data["tenure_group"] = pd.cut(
            months,
            bins=[-np.inf, 6, 12, 24, 36, 48, np.inf],
            labels=[
                "0-6 months",
                "7-12 months",
                "13-24 months",
                "25-36 months",
                "37-48 months",
                "Over 48 months",
            ],
        ).astype("object")

    # Replace infinite numeric values with missing values.
    numeric_columns = data.select_dtypes(include=np.number).columns
    data[numeric_columns] = data[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    return data
