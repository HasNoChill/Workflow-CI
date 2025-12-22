import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train = pd.read_csv("namadataset_preprocessing/X_train.csv")
X_test  = pd.read_csv("namadataset_preprocessing/X_test.csv")
y_train = pd.read_csv("namadataset_preprocessing/y_train.csv")
y_test  = pd.read_csv("namadataset_preprocessing/y_test.csv")

y_train = y_train.values.ravel()
y_test  = y_test.values.ravel()

mlflow.set_experiment("Customer_Churn_Experiment")
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
