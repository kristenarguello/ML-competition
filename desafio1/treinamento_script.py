# %%
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

SEED = 42

# %%
test_df = pd.read_csv("data/test.csv", index_col=[0])
train_df = pd.read_csv("data/train.csv", index_col=[0])

# %%
test_df.value_counts()

X = train_df.drop(columns=["class"])
y = train_df["class"]

# %%
# removes outliers with the iqr method
# by measuring the spread of the middle 50%
# removes anything that is out from that spread
# good since knn is sensitive to outliers
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
non_outliers = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
X = X[non_outliers]
y = y[X.index]

# %%

# PCA (retain 95% variance)
# this transforms features into new uncorrelated ones
# helps whith the problem of maldicao da dimensionalidade which can be a thing for knn
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=SEED
)

# %%
classifier = KNeighborsClassifier()
pipeline = Pipeline([("classifier", classifier)])

param_grid = {"classifier__n_neighbors": [3, 5, 7, 9]}

# divides the folds so there is an equal distribution of classes
# stratified k fold
# this is important since knn is sensitive to class imbalance
skf = StratifiedKFold(n_splits=10)
knn_grid = GridSearchCV(pipeline, param_grid, scoring="f1_weighted", cv=skf, n_jobs=-1)
knn_grid.fit(X_train, y_train)

# %%
# predict
test_df_pca = pca.transform(test_df)
final_predictions = knn_grid.predict(test_df_pca)


# %%
def create_submission_file(predictions, test_df, submission_file_name="submission.csv"):
    submission_df = pd.DataFrame({"id": test_df.index, "Target": predictions})
    submission_df.to_csv(submission_file_name, index=False)
    print(f"Submission file '{submission_file_name}' created successfully.")


create_submission_file(final_predictions, test_df)

# %%
