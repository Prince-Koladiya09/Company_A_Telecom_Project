def metrics_summary(metrics_row):
    
    return (
        f"The {metrics_row['Model']} baseline achieved ROC-AUC "
        f"{metrics_row['ROC_AUC']:.3f}, F1 {metrics_row['F1']:.3f}, "
        f"precision {metrics_row['Precision']:.3f}, and recall "
        f"{metrics_row['Recall']:.3f} on a held-out test set. "
        "These scores describe predictive performance on this split; they do not "
        "prove that a retention intervention will cause customers to stay."
    )

def business_summary(lift):

    return (
        f"The highest-risk {lift['target_share']:.0%} of the test set contained "
        f"{lift['selected_rows']:,} records. The observed positive rate in this "
        f"group was {lift['observed_positive_rate_top_group']:.1%}, compared with "
        f"{lift['observed_positive_rate_full_test']:.1%} in the full test set. "
        f"This group captured {lift['share_of_actual_positives_captured']:.1%} "
        "of positive cases in the test set. Treat this as retrospective targeting "
        "analysis, not evidence of causal campaign impact."
    )
