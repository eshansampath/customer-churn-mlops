import shap
import matplotlib.pyplot as plt
import os


def explain_model(model, X_train):
    print("Running SHAP explainability...")

    os.makedirs("models", exist_ok=True)

    try:
        # Convert to numeric
        X_train_shap = X_train.astype(float)

        # Sample data
        X_sample = X_train_shap.sample(300, random_state=42)

        # SHAP API 
        explainer = shap.TreeExplainer(model, X_sample)

        shap_values = explainer(X_sample)

        # Summary plot
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig("models/shap_summary.png", bbox_inches="tight")
        plt.close()

        # Bar plot (new API)
        shap.plots.bar(shap_values, show=False)
        plt.savefig("models/shap_bar.png", bbox_inches="tight")
        plt.close()

        print("SHAP plots saved successfully!")

    except Exception as e:
        print("SHAP ERROR:", e)