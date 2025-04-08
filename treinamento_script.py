# %%
from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np


# %%
test_df = pd.read_csv("data/test.csv", index_col=[0])
train_df = pd.read_csv("data/train.csv", index_col=[0])

# %%
test_df.value_counts()
# %%

X = train_df.drop(columns=["class"])
y = train_df["class"]

# %%
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
# %%

# Create a pipeline with standard scaling and decision tree

classifier = KNeighborsClassifier(n_neighbors=3)
pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=20, scoring="f1_macro")

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions, average="macro"))
print("Precision:", precision_score(y_test, predictions, average="macro"))
print("Recall:", recall_score(y_test, predictions, average="macro"))

final_predictions = pipeline.predict(test_df)


def create_submission_file(predictions, test_df, submission_file_name="submission.csv"):
    submission_df = pd.DataFrame({"id": test_df.index, "Target": predictions})
    submission_df.to_csv(submission_file_name, index=False)
    print(f"Submission file '{submission_file_name}' created successfully.")


create_submission_file(final_predictions, test_df)


# %%
classifier = DecisionTreeClassifier(random_state=SEED)
pipeline = Pipeline([("classifier", classifier)])

# Set up hyperparameters to tune
param_grid = {
    "classifier__criterion": [
        "gini",
        "entropy",
    ],  # or 'log_loss' for probabilistic output
    "classifier__max_depth": [
        None,
        3,
        5,
        7,
        10,
        13,
        15,
    ],  # Limit depth to avoid overfitting
    "classifier__min_samples_split": [
        2,
        5,
        10,
        20,
        50,
    ],  # Minimum samples needed to split a node
    "classifier__min_samples_leaf": [1, 2, 5, 10],  # Minimum samples in a leaf
    "classifier__max_features": [None, "sqrt", "log2"],  # Feature selection strategy
    "classifier__class_weight": [None, "balanced"],  # Adjust for class imbalance,\
    "classifier__splitter": ["best", "random"],  # Split strategy
}

# param_grid = {
#     "classifier__n_neighbors": [3, 5, 7],  # Number of neighbors
#     "classifier__weights": ["uniform", "distance"],  # Weighting strategy
#     "classifier__algorithm": [
#         "auto",
#         "ball_tree",
#         "kd_tree",
#         "brute",
#     ],  # Algorithm to use
#     "classifier__leaf_size": [
#         10,
#         20,
#         30,
#         40,
#         50,
#     ],  # Leaf size for tree-based algorithms
#     "classifier__metric": ["euclidean", "manhattan", "minkowski"],  # Distance metric
#     "classifier__metric_params": [None],  # Additional metric parameters
#     "classifier__n_jobs": [-1],  # Use all available cores
#     "classifier__p": [1, 2],  # Power parameter for Minkowski distance
# }

grid_search = GridSearchCV(pipeline, param_grid, cv=20, scoring="f1_macro", n_jobs=-1)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=20,
    scoring="f1_macro",
)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_

# %%
model.fit(X_train, y_train)

# %%
predictions = model.predict(X_test)


# %%
print("Accuracy:", accuracy_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions, average="macro"))
print("Precision:", precision_score(y_test, predictions, average="macro"))
print("Recall:", recall_score(y_test, predictions, average="macro"))
# %%
final_predictions = model.predict(test_df)


# %%
# def create_submission_file(predictions, test_df, submission_file_name="submission.csv"):
#     submission_df = pd.DataFrame({"id": test_df.index, "Target": predictions})
#     submission_df.to_csv(submission_file_name, index=False)
#     print(f"Submission file '{submission_file_name}' created successfully.")


# %%
create_submission_file(final_predictions, test_df)
# %%


# naive bayes
# Create a pipeline with standard scaling and Naive Bayes
classifier = GaussianNB()

pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])

# Set up hyperparameters to tune for Naive Bayes
param_grid = {
    "classifier__var_smoothing": np.logspace(-10, -1, 10),  # Smoothing variance
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1
)

cv_scores = cross_val_score(
    pipeline, X_train, y_train, cv=5, scoring="balanced_accuracy"
)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_

# Train and evaluate
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions, average="macro"))
print("Precision:", precision_score(y_test, predictions, average="macro"))
print("Recall:", recall_score(y_test, predictions, average="macro"))

# Make predictions on test data
final_predictions = model.predict(test_df)

# Create submission file
create_submission_file(final_predictions, test_df, "submission_naive_bayes.csv")
# %%
