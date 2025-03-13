import mlflow
import mlflow.sklearn

# Ensure MLflow server is reachable
mlflow.set_tracking_uri("http://0.0.0.0:5000")

# Set experiment (creates if missing)
experiment_name = "Iris_Classification"
mlflow.set_experiment(experiment_name)

# Start a new MLflow run
with mlflow.start_run():  
    run_id = mlflow.active_run().info.run_id 
    model_uri = f"runs:/{run_id}/model"

    # Register the model
    mlflow.register_model(model_uri, "IrisModel")

print("Model registered successfully!")
