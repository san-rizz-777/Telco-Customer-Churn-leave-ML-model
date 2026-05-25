import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import recall_score

"""
Testing the second phase of pipeline -> Data modeling , hyperparameter tuning and evaluation.
"""


print("Phase 2 of pipeline testing is starting.......")

df = pd.read_csv("data/processed_data/telco_churn_processed.csv")

# The target must be numeric
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No":0, "Yes":1})

assert df["Churn"].isna().sum() == 0, "Churn has NaNs"
assert set(df["Churn"].unique()) <= {0,1}, "Churn is not 0/1"

# Extracting the features and label
X = df.drop(columns=['Churn'])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,  random_state=42)

THRESHOLD=0.3

# Objective function for tuning hyperparameters - (using recall score)
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    return recall_score(y_test, y_pred, pos_label=1)

# Create a study by optuna and optimize by objectrive function
# Return the best params and recall
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(f"Best params:- {study.best_params}")
print(f"Best recall:- {study.best_value}")

print("Phase 2 of pipeline testing completed succesfully!!!!")