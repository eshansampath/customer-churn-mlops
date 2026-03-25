from sklearn.metrics import (
    classification_report, accuracy_score, roc_curve, auc,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import numpy as np

from mlops.mlflow_tracking import log_experiment


def evaluate(model, X_test, y_test, X_train, params):
    # Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Threshold tuning (F1)
    thresholds = np.arange(0.3, 0.8, 0.05)

    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        preds = (y_prob > t).astype(int)
        f1 = f1_score(y_test, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"Best Threshold: {best_threshold:.2f} | F1: {best_f1:.3f}")

    y_pred = (y_prob > best_threshold).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nFinal Evaluation")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig("models/roc_curve.png")
    plt.close()

    # MLflow logging
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "best_threshold": best_threshold
    }

    artifacts = [
        "models/roc_curve.png"
    ]

    log_experiment(params, metrics, artifacts)

    return acc, roc_auc