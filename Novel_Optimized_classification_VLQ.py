import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report, make_scorer
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from matplotlib import pyplot as plt
import itertools
import seaborn as sns
import time
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings("ignore")


# Read pelvic.csv
data = pd.read_csv("pelvic.csv")

# Prepare features and target
X = data.drop("VLQ", axis=1)
y = data["VLQ"]

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

# Model 1: Random Forest Classifier with Optimized Max Depth
rf = RandomForestClassifier(n_estimators=100, random_state=42)
param_grid = {"max_depth": [3, 4, None]}
grid_search = GridSearchCV(rf, param_grid, scoring=make_scorer(f1_score, average='macro'), cv=StratifiedShuffleSplit(test_size=0.2, random_state=42))
start_time = time.time()
grid_search.fit(X, y)
end_time = time.time()
runtime_rf = end_time - start_time
rf_optimal = grid_search.best_estimator_
rf_preds = rf_optimal.predict(X)
print("Random Forest Classifier with optimized max depth:")
print(classification_report(y, rf_preds))
print(f"Execution Time: {runtime_rf:.4f} seconds")

# Model 2: Extra Trees Classifier
etc = ExtraTreesClassifier(n_estimators=100, max_depth=None, random_state=42)
start_time = time.time()
etc.fit(X, y)
end_time = time.time()
runtime_etc = end_time - start_time
etc_preds = etc.predict(X)
print("Extra Trees Classifier:")
print(classification_report(y, etc_preds))
print(f"Execution Time: {runtime_etc:.4f} seconds")

# Model 3: Gradient Boosting Classifier with Dimensionality Reduction
pipe_gb = Pipeline([
    ("pca", PCA()),
    ("gb", GradientBoostingClassifier(random_state=42))
])

# Reduced parameter grid
param_grid_gb = {
    "pca__n_components": [1, 2],  # Reduced number of components for faster computation
    "gb__learning_rate": [0.01, 0.1],
    "gb__max_depth": [3, 5]
}

# Randomized search with reduced CV folds
random_search_gb = RandomizedSearchCV(pipe_gb, param_distributions=param_grid_gb, 
                                      scoring=make_scorer(f1_score, average='macro'), 
                                      cv=StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
                                      n_iter=5)  # Number of parameter settings that are sampled
start_time = time.time()
random_search_gb.fit(X, y)
end_time = time.time()
runtime_gb = end_time - start_time
gb_optimal = random_search_gb.best_estimator_
gb_preds = gb_optimal.predict(X)
print("Gradient Boosting Classifier with dimensionality reduction:")
print(classification_report(y, gb_preds))
print(f"Execution Time: {runtime_gb:.4f} seconds")

# Model 4: Voting Classifier
lr = LogisticRegression(random_state=42)
rf_hard = RandomForestClassifier(random_state=42)
etc_hard = ExtraTreesClassifier(random_state=42)
vote_clf = VotingClassifier(estimators=[("lr", lr), ("rf", rf_hard), ("et", etc_hard)], voting="hard")
start_time = time.time()
vote_clf.fit(X, y)
end_time = time.time()
runtime_vc = end_time - start_time
vote_preds = vote_clf.predict(X)
pos_classes = unique_labels(y)
total_auc = 0

for pos_class in pos_classes:
    neg_class = [cls for cls in unique_labels(y) if cls != pos_class][0]
    print("Voting Classifier:")
    print(classification_report(y, vote_preds))  # Removed the target_names argument
    draw_roc(y, vote_preds, "Voting Classifier", pos_class)
    _, _, roc_auc = roc_curve(y == pos_class, vote_preds)
    total_auc += roc_auc.sum()

print("Aggregate AUC Score: {:.2f}".format(total_auc))
img_confusion_matrix(confusion_matrix(y, vote_preds), ['0', '1'])
print(f"Execution Time: {runtime_vc:.4f} seconds")