from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the XGBoost model on test set.

    Args:
        model : The trained model.
        X_test : The test features.
        y_test : The test labels.
    """

    y_pred = model.predict(X_test)
    print("Classification report:- \n", classification_report(y_test, y_pred))
    print("Confusion matrix:- \n", confusion_matrix(y_test, y_pred))