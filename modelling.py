import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set MLflow URI dan eksperimen
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Obesity Modeling - SVM")

# Aktifkan autologging untuk scikit-learn
mlflow.autolog()

# Load dataset
X_train = pd.read_csv("data_preprocessing/X_train.csv")
X_test = pd.read_csv("data_preprocessing/X_test.csv")
y_train = pd.read_csv("data_preprocessing/y_train.csv")
y_test = pd.read_csv("data_preprocessing/y_test.csv")

# Konversi y_train dan y_test ke Series
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Mulai MLflow run
with mlflow.start_run(run_name="SVM_Model"):
    # Latih model
    model = SVC()
    model.fit(X_train, y_train)

    # Evaluasi model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Autolog sudah mencatat metrik dan model, jadi print saja hasilnya
    print(f"Akurasi model (default SVC): {accuracy:.4f}")
