from dagster import asset
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import mlflow

@asset(
    description="Evaluates trained model and logs metrics to MLflow.",
    compute_kind="ml",
    required_resource_keys={"mlflow"},
)
def model_evaluation(context, split_data, trained_model):
    """
    Evaluates the trained model using test data.
    Returns a dictionary of evaluation metrics.
    """
    
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    preds = trained_model.predict(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, preds)
    fbeta = fbeta_score(y_test, preds, beta=2)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    # Log to MLflow
    mlflow = context.resources.mlflow
    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("fbeta", fbeta)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_dict(report, "classification_report.json")

    return {
        "accuracy": acc,
        "fbeta": fbeta,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
