import mlflow
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import random
import numpy as np
 
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
 
# Create a new MLflow Experiment
mlflow.set_experiment("Obesity Modeling - Hyperparameter Tuning SVM")
 
X_train = pd.read_csv("data_preprocessing/X_train.csv")
X_test = pd.read_csv("data_preprocessing/X_test.csv")
y_train = pd.read_csv("data_preprocessing/y_train.csv").squeeze()
y_test = pd.read_csv("data_preprocessing/y_test.csv").squeeze()
 

input_example = X_train[0:5]

# Range hyperparameter yang akan dicoba
C_range = np.logspace(-2, 2, 5)  # 5 nilai dari 0.01 sampai 100 (log scale)
kernel_options = ['linear', 'rbf', 'poly']
gamma_range = ['scale', 'auto']  # bisa juga coba beberapa float tapi ini default aja

best_accuracy = 0
best_params = {}

for C in C_range:
    for kernel in kernel_options:
        for gamma in gamma_range:
            run_name = f"SVC_C{C}_kernel{kernel}_gamma{gamma}"
            with mlflow.start_run(run_name="SVM_Tuning_Model"):
                mlflow.autolog()  # otomatis log parameter dan model
                
                # Inisiasi dan train model
                model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluasi
                accuracy = model.score(X_test, y_test)
                mlflow.log_metric("accuracy", accuracy)
                
                # Simpan model terbaik secara eksplisit
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {"C": C, "kernel": kernel, "gamma": gamma}
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        input_example=input_example
                    )

print("Best Accuracy:", best_accuracy)
print("Best Params:", best_params)  