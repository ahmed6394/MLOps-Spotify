from dagster import asset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

@asset(
    description="Trains ML model using MLFlow tracking",
    required_resource_keys={"mlflow"},
    compute_kind="ml"
)
def trained_model(split_data, mlflow):
    """
    Trains a RandomForest model using the split data and logs metrics to MLflow.
    Returns the trained model.
    """
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    # Define the model pipeline
    model = Pipeline(
        steps=[
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
        ]
    )

    # Mlflow tracking
    with mlflow.start_run(run_name="rf_model_training"):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        fbeta = fbeta_score(y_test, preds, beta=2)

        #log matrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("fbeta", fbeta)

        # log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # log params
        mlflow.log_param({"n_estimators": 200, "random_state": 42})

        print(f"Accuracy: {acc:.4f}, Fbeta: {fbeta:.4f}")


    return model
