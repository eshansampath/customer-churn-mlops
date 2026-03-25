import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn_prediction")


def log_experiment(params, metrics, artifacts=None):
    """
    params: dict
    metrics: dict
    artifacts: list of file paths
    """
    with mlflow.start_run():

        # Log parameters
        if params:
            mlflow.log_params(params)

        # Log metrics
        if metrics:
            mlflow.log_metrics(metrics)

        # ROC curve
        try:
            mlflow.log_artifact("models/roc_curve.png")
        except Exception:
            print("ROC curve not found, skipping...")

        # Log additional artifacts (SHAP)
        if artifacts:
            for path in artifacts:
                try:
                    mlflow.log_artifact(path)
                except Exception:
                    print(f"Could not log artifact: {path}")