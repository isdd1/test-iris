import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
iris = pd.read_csv('iris.csv')
X = iris.drop(columns=['Id', 'Species'])
y = iris['Species']

# Load model
model = joblib.load("iris_model.pkl")

# Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)  # Jenkins reads this output
