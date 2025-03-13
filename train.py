import mlflow
import mlflow.sklearn
from iris_model import create_model, save_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Enable MLflow autologging
mlflow.sklearn.autolog()

if __name__ == "__main__":
    # Load and split the dataset
    iris = pd.read_csv('iris.csv')
    X = iris.drop(columns=['Id', 'Species'])
    y = iris['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model, _, _ = create_model()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics manually
        mlflow.log_metric("accuracy", accuracy)

        # Save and log the model
        save_model(model, "iris_model.pkl")
        mlflow.sklearn.log_model(model, "model")

    print(f"Model trained and logged with accuracy: {accuracy:.4f}")
