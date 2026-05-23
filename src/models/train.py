import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(df: pd.DataFrame, target_col:str):
    """
    Trains a XGBoost model and logs it with MLflow

    Args:
    df :- feature dataframe
    target_col :- (str) name of target column
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ## xgboost estimator
    xgb = XGBClassifier(n_estimators=600, learning_rate=0.0123, max_depth=3, subsample=0.999, colsample_bytree=0.791,
    min_child_weight=9, gamma=4.982, reg_alpha=0.727, reg_lambda=4.268)

    with mlflow.start_run():
        # Train the model
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        # Log parametrs, metrics and model
        mlflow.log_param("n_estimators", 600)
        mlflow.log_metric("rec", rec)
        mlflow.log_metric("acc", acc)
        mlflow.xgboost.log_model(xgb, "model")

        # Log the dataset
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")