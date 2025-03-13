from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas
import joblib

def read_csv(file_path):
    data = pandas.read_csv(file_path)
    return data

def create_model():
    iris = read_csv('iris.csv')
    X = iris.drop(columns=['Id', 'Species'])
    y = iris['Species']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with StandardScaler and LogisticRegression
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    return model, X_train, y_train

def save_model(model, filename='iris_model.pkl'):
    joblib.dump(model, filename)
