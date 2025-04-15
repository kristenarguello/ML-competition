# %%
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer  # , TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


# %%
def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()


# %%

train_df = pd.read_csv("data/train.csv", index_col="id")
test_df = pd.read_csv("data/test.csv", index_col="id")

# %%
train_df["clean_review"] = train_df["review"]  # .apply(remove_html)
test_df["clean_review"] = test_df["review"]  # .apply(remove_html)

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train_df["clean_review"], train_df["label"], test_size=0.2, random_state=42
)
# %%


pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        # ("tfidf", TfidfTransformer()),
        ("clf", DecisionTreeClassifier(random_state=42)),
    ]
)

# %%
param_grid = {
    "clf__criterion": ["log_loss"],
    "clf__ccp_alpha": [0.001],
    "clf__max_depth": [None],
}


grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
# %%

print("Melhores parâmetros encontrados:", grid_search.best_params_)
print("Melhor acurácia na validação cruzada:", grid_search.best_score_)
# %%


y_pred_val = grid_search.predict(X_val)
print("Acurácia no conjunto de validação:", accuracy_score(y_val, y_pred_val))
# %%


y_pred_test = grid_search.predict(test_df["clean_review"])
submission_df = pd.DataFrame({"id": test_df.index, "Target": y_pred_test})
submission_df.to_csv("submission.csv", index=False)
print("Arquivo 'submission.csv' criado com sucesso!")
# %%
