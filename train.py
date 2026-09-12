from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas
import os
import json
import joblib

def main():
    iris = load_iris()
    X, Y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(model, model_path)

    accuracy = model.score(X_test, y_test)
    metrics = {"accuracy": float(accuracy)}
    print(f'Test accuracy is: {metrics}')


if __name__ == "__main__":
    main()