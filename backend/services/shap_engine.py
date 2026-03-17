import os
import shap
import matplotlib.pyplot as plt


def generate_shap_plots(model, X, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
    except:
        explainer = shap.Explainer(model, X)

    shap_values = explainer(X)

    plots = {}

    # Summary plot
    summary_path = os.path.join(save_dir, "shap_summary.png")

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()

    plots["shap_summary"] = summary_path

    # Feature importance (bar)
    bar_path = os.path.join(save_dir, "shap_feature_importance.png")

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()

    plots["shap_importance"] = bar_path

    return plots