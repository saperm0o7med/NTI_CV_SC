# 🌸 Iris Flower Classification App

Welcome to my **first project** completed during the **NTI Summer Training Program**, specializing in **Computer Vision**! This repository features a complete, end-to-end Machine Learning web application that predicts iris flower species based on their morphological measurements.

---

##  Project Overview
The **Iris Flower Classification App** is built using a Deep Learning model trained with **TensorFlow/Keras**. To make the model accessible and interactive, I designed a clean and responsive web interface using **Streamlit**. The app takes four key feature inputs, processes them through a pre-fitted scaler, and outputs the predicted Iris species in real-time.

### Features Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Web Interface:** Streamlit
* **Data Processing & Scaling:** Scikit-learn (`StandardScaler` saved via `pickle`)
* **Scientific Computing:** NumPy & Pandas

---

## 🛠️ Project Structure
```text
1stProject/
│
├── venv/                 # Virtual Environment (Ignored in Git)
├── App.py                # Main Streamlit web application source code
├── Load_iris.keras       # Pre-trained Keras Deep Learning model
├── scaler.pkl            # Serialized scaler for input feature normalization
├── requirements.txt      # Project dependencies and libraries
└── .gitignore            # Specifying untracked files to ignore
