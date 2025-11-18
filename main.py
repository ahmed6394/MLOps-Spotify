# check_artifacts.py
import mlflow
from mlflow.tracking import MlflowClient

def check_run_artifacts():
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    run_id = "20e4a950ac0d404d939a07e11eab1e86"
    
    try:
        # List all artifacts in the run
        artifacts = client.list_artifacts(run_id)
        print("Artifacts in run:")
        for artifact in artifacts:
            print(f"  - {artifact.path} (dir: {artifact.is_dir})")
            
            # If it's a directory, list its contents
            if artifact.is_dir:
                sub_artifacts = client.list_artifacts(run_id, artifact.path)
                for sub_artifact in sub_artifacts:
                    print(f"    * {sub_artifact.path}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_run_artifacts()