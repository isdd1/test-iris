import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("Iris_Classification")

model_uri = "runs:/{}/model".format(mlflow.active_run().info.run_id)
mlflow.register_model(model_uri, "IrisModel")

print("Model registered successfully!")
