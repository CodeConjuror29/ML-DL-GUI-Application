import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, r2_score, confusion_matrix,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML-DL GUI Project - Weeks 3 to 13")

        self.df = None
        self.algorithm = tk.StringVar(value="Logistic Regression")
        self.last_results = None

        # Buttons
        tk.Button(root, text="Load CSV", command=self.load_csv).pack(pady=5)

        # Dataset Preview
        self.tree = ttk.Treeview(root)
        self.tree.pack(pady=5, fill="both", expand=True)

        # Algorithm Selection
        tk.Label(root, text="Select Algorithm:").pack(pady=5)
        algo_menu = tk.OptionMenu(root, self.algorithm,
            "Logistic Regression",
            "Linear Regression",
            "Decision Tree",
            "kNN",
            "SVM",
            "Random Forest",
            "Naive Bayes",
            "Gradient Boosting",
            "Neural Network"
        )
        algo_menu.pack(pady=5)

        # Checkboxes
        self.cv_var = tk.BooleanVar()
        self.pca_var = tk.BooleanVar()
        tk.Checkbutton(root, text="Run Cross-Validation", variable=self.cv_var).pack(pady=3)
        tk.Checkbutton(root, text="Apply PCA (2 Components)", variable=self.pca_var).pack(pady=3)

        # Run / Clear / Save Buttons
        tk.Button(root, text="Run Model", command=self.run_model).pack(pady=8)
        tk.Button(root, text="Clear Output", command=self.clear_output).pack(pady=3)
        tk.Button(root, text="Save Results", command=self.save_results).pack(pady=3)

        # Output Label
        self.output_label = tk.Label(root, text="", font=("Arial", 12), justify="left")
        self.output_label.pack(pady=10)

        # Status Label
        self.status_label = tk.Label(root, text="", fg="blue")
        self.status_label.pack(pady=3)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        self.df = pd.read_csv(file_path)

        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree["column"] = list(self.df.columns)
        self.tree["show"] = "headings"

        for col in self.df.columns:
            self.tree.heading(col, text=col)

        for row in self.df.head(20).values.tolist():
            self.tree.insert("", "end", values=row)

        messagebox.showinfo("Loaded", "Dataset loaded successfully!")

    def clear_output(self):
        self.output_label.config(text="")
        self.status_label.config(text="")
        self.last_results = None

    def run_model(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return

        try:
            self.status_label.config(text="Training in progress...")
            self.root.update_idletasks()

            X = self.df.iloc[:, :-1]
            y = self.df.iloc[:, -1]

            if y.dtype == "object" or y.dtype.name == "category":
                le = LabelEncoder()
                y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # PCA (if selected)
            if self.pca_var.get():
                pca = PCA(n_components=2)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            algo = self.algorithm.get()
            model = None
            param_grid = None
            is_classification = True

            # Choose model + params
            if algo == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif algo == "Linear Regression":
                model = LinearRegression()
                is_classification = False
            elif algo == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
                param_grid = {"max_depth": [3, 5, 7, None]}
            elif algo == "kNN":
                model = KNeighborsClassifier()
                param_grid = {"n_neighbors": [3, 5, 7]}
            elif algo == "SVM":
                model = SVC()
                param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
            elif algo == "Random Forest":
                model = RandomForestClassifier(random_state=42)
                param_grid = {"n_estimators": [50, 100, 150]}
            elif algo == "Naive Bayes":
                model = GaussianNB()
            elif algo == "Gradient Boosting":
                model = GradientBoostingClassifier()
            elif algo == "Neural Network":
                model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
                param_grid = {"hidden_layer_sizes": [(50,), (100,), (100, 50)]}

            # Grid search if available
            if param_grid:
                grid = GridSearchCV(model, param_grid, cv=3)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                best_params_text = f"Best Params: {grid.best_params_}\n"
            else:
                model.fit(X_train, y_train)
                best_params_text = ""

            y_pred = model.predict(X_test)
            results_text = best_params_text

            # Evaluate
            if is_classification:
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                results_text += f"Accuracy: {acc:.2f}\nPrecision: {prec:.2f}\nRecall: {rec:.2f}\nF1-score: {f1:.2f}"

                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"{algo} Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.show()
            else:
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                results_text += f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}"

            # Cross-validation
            if self.cv_var.get():
                cv_scores = cross_val_score(model, X, y, cv=5)
                results_text += f"\nCV Mean Score: {cv_scores.mean():.2f}"

            self.output_label.config(text=results_text)
            self.last_results = results_text
            self.status_label.config(text="Training completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Error occurred during training.")

    def save_results(self):
        if not self.last_results:
            messagebox.showwarning("No Results", "Run a model first!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                for line in self.last_results.split("\n"):
                    writer.writerow([line])
            messagebox.showinfo("Saved", "Results saved successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
