import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report, make_scorer
from sklearn.utils.multiclass import unique_labels
from matplotlib import pyplot as plt
import itertools
import seaborn as sns
from sklearn.svm import LinearSVC
import time
import warnings
warnings.filterwarnings("ignore")


# Read pelvic.csv
data = pd.read_csv("pelvic.csv")

# Prepare features and target
X = data.drop("Qxford scale", axis=1)
y = data["Qxford scale"]

def draw_roc(y_true, y_pred, model_name, pos_class):
    fpr, tpr, _ = roc_curve(y_true == pos_class, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='{} {}'.format(model_name, pos_class) + ' (AUC = '+ str(round(roc_auc, 2)) + ')')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.title('Receiver Operating Characteristics (ROC) curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.show()

def img_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    plt.title('Confusion Matrix')
    plt.show()

# Model 1: Decision Tree Classifier with Optimal Max Depth
dt = DecisionTreeClassifier(random_state=42)
param_grid = {"max_depth": [3, 4, None]}
grid_search = GridSearchCV(dt, param_grid, scoring=make_scorer(f1_score, average='macro'), cv=StratifiedShuffleSplit(test_size=0.2, random_state=42))
start_time = time.time()
grid_search.fit(X, y)
end_time = time.time()
runtime_dt = end_time - start_time
dt_optimal = grid_search.best_estimator_
dt_preds = dt_optimal.predict(X)
print("Decision Tree Classifier with optimum max depth:")
print(classification_report(y, dt_preds))
print(f"Execution Time: {runtime_dt:.4f} seconds")

# Model 2: Logistic Regression
start_time = time.time()
lr = LogisticRegression(random_state=42)
lr.fit(X, y)
end_time = time.time()
runtime_lr = end_time - start_time
lr_preds = lr.predict(X)
print("Logistic Regression:")
print(classification_report(y, lr_preds))
print(f"Execution Time: {runtime_lr:.4f} seconds")

# Model 3: Random Forest Classifier with Hyperparameters Tuning
rf = RandomForestClassifier(n_estimators=100, random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_features": [None, "auto", "sqrt"],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}
grid_search = GridSearchCV(rf, param_grid, scoring=make_scorer(f1_score, average='macro'), cv=StratifiedShuffleSplit(test_size=0.2, random_state=42))
start_time = time.time()
grid_search.fit(X, y)
end_time = time.time()
runtime_rf = end_time - start_time
rf_optimal = grid_search.best_estimator_
rf_preds = rf_optimal.predict(X)
print("Random Forest Classifier with hyperparameters tuning:")
print(classification_report(y, rf_preds))
print(f"Execution Time: {runtime_rf:.4f} seconds")

# Model 4: Support Vector Machine
start_time = time.time()
svc = LinearSVC(random_state=42)
svc.fit(X, y)
end_time = time.time()
runtime_svc = end_time - start_time
svc_preds = svc.predict(X)
print("Support Vector Machine (LinearSVC):")
print(classification_report(y, svc_preds))
print(f"Execution Time: {runtime_svc:.4f} seconds")