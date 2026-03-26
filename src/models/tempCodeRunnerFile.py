shap.summary_plot(
        shap_array,
        X_scaled,
        feature_names=FEATURE_NAMES,
        plot_type="bar"
    )