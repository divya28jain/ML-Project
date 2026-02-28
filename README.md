# End-to-End Machine Learning Project with MLOps (MLflow + DagsHub)

This project is a production-ready End-to-End Machine Learning pipeline built following Krish Naik’s structured Data Science workflow and enhanced with full MLOps integration using MLflow and DagsHub Model Registry.

It covers the complete lifecycle of a machine learning system — from data ingestion to model versioning.

---

## 🚀 Project Overview

This project builds a regression model to predict student performance scores using multiple machine learning algorithms.

The pipeline automatically:

- Ingests raw data
- Performs data transformation
- Trains multiple ML models
- Performs hyperparameter tuning
- Selects the best model
- Logs experiments using MLflow
- Registers model versions in DagsHub
- Saves trained artifacts locally

---

## 🏗 Project Architecture

ML_PROJECT/
│
├── artifacts/ # Saved trained models & outputs
├── logs/ # Logging files
├── mlruns/ # MLflow local tracking
│
├── src/ml_project/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ ├── model_trainer.py
│ │ ├── model_monitoring.py
│ │
│ ├── pipelines/
│ ├── utils.py
│ ├── logger.py
│ ├── exception.py
│ ├── init.py
│
├── notebook/ # EDA & transformation notebooks
├── app.py # Main pipeline runner
├── main.py
├── Dockerfile
├── requirements.txt
├── setup.py
└── README.md

---

## 🧠 Models Compared

The pipeline trains and compares:

- Linear Regression
- Random Forest
- Decision Tree
- Gradient Boosting
- AdaBoost
- XGBoost
- CatBoost

The best model is selected automatically based on **R² Score**.

---

## 📊 Example Model Performance
Best Model: Linear Regression
R2 Score: 0.88
RMSE: 5.39
MAE: 4.21

---

## 🔬 Experiment Tracking (MLflow)

This project integrates MLflow to log:

- Hyperparameters
- Evaluation metrics
- Model artifacts
- Run metadata

All experiments are tracked remotely using **DagsHub MLflow backend**.

You can view experiments at:

👉 https://dagshub.com/divya28jain/ML-Project.mlflow

---

## 📦 Model Registry (DagsHub)

After training, the best model is:

- Registered in DagsHub Model Registry
- Version controlled
- Linked with experiment run
- Stored as an artifact

Example:
Model Name: Linear Regression
Version: 1


---

## ⚙️ Technologies Used

- Python
- Scikit-learn
- XGBoost
- CatBoost
- MLflow
- DagsHub
- NumPy
- Pandas
- Logging & Custom Exception Handling

---

## 🛠 How To Run This Project

### 1️⃣ Clone Repository
git clone https://github.com/divya28jain/ML-Project.git
cd ML-Project

---

### 2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

---

### 3️⃣ Install Dependencies
pip install -r requirements.txt

---

### 4️⃣ Set DagsHub Credentials

---

### 5️⃣ Run The Pipeline
python app.py

---

## 📁 Output

- Best model saved in `artifacts/model.pkl`
- Experiment logged in DagsHub
- Model registered in Model Registry
- Metrics printed in terminal

---

## 🌟 Key Highlights

✔ Structured production-level ML project  
✔ Automated model comparison  
✔ Hyperparameter tuning  
✔ Custom logging & exception handling  
✔ MLflow experiment tracking  
✔ Remote model registry  
✔ Model versioning  
✔ Reproducible training pipeline  

---

## 👩‍💻 Author

Divya Jain  
B.Tech Computer Science  
Machine Learning & MLOps Enthusiast  

---

⭐ If you found this project useful, consider giving it a star!