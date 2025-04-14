# %%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# %%
train_df = pd.read_csv("data/train.csv", index_col="id")
test_df = pd.read_csv("data/test.csv", index_col="id")

# %%

train_df
# %%
X_train, X_test, y_train, y_test = train_test_split(
    train_df["review"], train_df["label"], test_size=0.2, random_state=42
)
# %%
cv = CountVectorizer()
cv.fit(X_train)

# %%
X_train_cv = cv.transform(X_train)

# %%
X_train_cv
# %%
X_train_cv_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names_out())
X_train_cv_df

# %%
model = MultinomialNB()

# %%
model.fit(X_train_cv, y_train)

# %%
X_test_cv = cv.transform(X_test)
# %%
y_pred = model.predict(X_test_cv)
# %%
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# %%
def create_submission_file(predictions, test_df, submission_file_name="submission.csv"):
    submission_df = pd.DataFrame({"id": test_df.index, "Target": predictions})
    submission_df.to_csv(submission_file_name, index=False)
    print(f"Submission file '{submission_file_name}' created successfully.")


# %%
create_submission_file(
    model.predict(cv.transform(test_df["review"])), test_df, "submission.csv"
)

# %%
