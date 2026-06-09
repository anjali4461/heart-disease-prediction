
# Heart Disease Prediction using KNN

## 📌 Overview

This project is a **Heart Disease Prediction Web Application** that predicts whether a person is at risk of heart disease based on medical parameters. Multiple machine learning algorithms were evaluated, and **K-Nearest Neighbors (KNN)** achieved the best performance, making it the final model used in the application.

The web application provides an easy-to-use interface where users can enter patient health data and receive a prediction instantly.

---

## 🚀 Features

* Predicts the likelihood of heart disease.
* User-friendly web interface.
* Real-time prediction using a trained KNN model.
* Data preprocessing and feature scaling.
* Machine Learning model evaluation and comparison.
* Fast and lightweight deployment.

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* Matplotlib
* Seaborn
* Joblib

---

## 📊 Dataset

The model was trained using a heart disease dataset containing various medical attributes such as:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG Results
* Maximum Heart Rate Achieved
* Exercise-Induced Angina
* ST Depression
* Slope of Peak Exercise ST Segment
* Number of Major Vessels
* Thalassemia

**Target Variable:**

* 0 → No Heart Disease
* 1 → Heart Disease

---

## 🤖 Machine Learning Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Feature Scaling
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Selection of Best Model (KNN)
8. Web Application Integration

---

## 📈 Model Performance

Several machine learning algorithms were tested and compared.

| Model               | Accuracy       |
| ------------------- | -------------- |
| Logistic Regression | 86%            |
| Decision Tree       | 77%            |
|  Naive Bayes        | 85%            |
| SVM                 | 84%            |
| KNN                 | **86% (Best)** |

----

## 🔮 Future Improvements

* Deploy on cloud platforms (Render, Railway, AWS, etc.)
* Add more advanced ML/DL models
* Improve UI/UX design
* Provide risk percentage and visual analytics
* Integrate patient history tracking

website link : "https://heart-disease-prediction-sgjxyrcpsbdginjyugutfq.streamlit.app/"
