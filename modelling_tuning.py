import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import os
from dotenv import load_dotenv

# Load .env dan ambil username/password
load_dotenv()
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not username or not password:
    raise EnvironmentError("MLFLOW_TRACKING_USERNAME dan MLFLOW_TRACKING_PASSWORD harus di-set sebagai environment variable")

# Set environment variable untuk autentikasi Basic Auth MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = password

# Set URI tracking ke DagsHub kamu
mlflow.set_tracking_uri("https://dagshub.com/amvstofa/Membangun-model.mlflow")

# Set experiment name
mlflow.set_experiment("Obesity Modeling - Hyperparameter Tuning SVM")

# Load data
X_train = pd.read_csv("data_preprocessing/X_train.csv")
X_test = pd.read_csv("data_preprocessing/X_test.csv")
y_train = pd.read_csv("data_preprocessing/y_train.csv").squeeze()
y_test = pd.read_csv("data_preprocessing/y_test.csv").squeeze()

input_example = X_train.head(5)

# Hyperparameter grid
C_range = np.logspace(-2, 2, 5)
kernel_options = ['linear', 'rbf', 'poly']
gamma_range = ['scale', 'auto']

best_accuracy = 0
best_params = {}

for C in C_range:
    for kernel in kernel_options:
        for gamma in gamma_range:
            run_name = f"SVC_C{C}_kernel{kernel}_gamma{gamma}"
            with mlflow.start_run(run_name=run_name):
                mlflow.sklearn.autolog()
                
                model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
                model.fit(X_train, y_train)
                
                accuracy = model.score(X_test, y_test)
                mlflow.log_metric("accuracy", accuracy)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {"C": C, "kernel": kernel, "gamma": gamma}
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="best_model",
                        input_example=input_example
                    )

print("Best Accuracy:", best_accuracy)
print("Best Params:", best_params)
