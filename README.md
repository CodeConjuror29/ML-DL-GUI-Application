ML-DL GUI Application

📘 Overview  
This project is a **GUI-based Machine Learning and Deep Learning toolkit** developed as part of my research internship under the guidance of **Dr. Palash Dey**, Assistant Professor, Department of Computer Science and Engineering, IIT Kharagpur.  

The aim of this project was to build an **interactive Windows application** that allows users to **load datasets, apply popular ML/DL algorithms, evaluate performance, and visualize results** — all through an easy-to-use interface.

---

🧩 Description  
This application provides a simple and visual way to perform machine learning experiments without writing code.  
Users can load CSV datasets, choose different ML or DL algorithms, apply PCA for dimensionality reduction, tune hyperparameters, and view model performance instantly.  
It serves as a compact educational tool for understanding how different algorithms perform on various datasets, with clear metrics and graphical outputs.

---

⚙️ Features
- Load and preview datasets (CSV format)  
- Choose from multiple **Machine Learning algorithms**:
  - Logistic Regression  
  - Linear Regression  
  - Decision Tree  
  - k-Nearest Neighbors (kNN)  
  - Support Vector Machine (SVM)  
  - Random Forest  
  - Naive Bayes  
  - Gradient Boosting  
  - Neural Network (MLPClassifier)  
- Optional **PCA (Principal Component Analysis)** for dimensionality reduction  
- **Cross-validation (K-Fold)** and **GridSearchCV** for hyperparameter tuning  
- Model **evaluation metrics**:
  - Accuracy, Precision, Recall, F1-score (Classification)  
  - MAE, MSE, RMSE, R² (Regression)  
- **Confusion Matrix visualization**  
- **Result export** to CSV  
- Clear Output and Status updates during model training  

---

🧠 Technologies Used
- **Python 3**  
- **Tkinter** – GUI development  
- **Scikit-learn** – Machine Learning  
- **Pandas, NumPy** – Data handling  
- **Matplotlib, Seaborn** – Visualizations  

---

📊 Sample Datasets
Tested on:  
- **Iris Dataset** (Classification)  
- **Housing Dataset** (Regression)

---

🚀 Future Scope
- Extend support for **image datasets (medical image analysis)**  
- Add **CNN models (using TensorFlow/Keras)** for deep learning tasks  
- Include more advanced visualizations (ROC, AUC, feature importance)  
- Deploy as a standalone executable (.exe)  
