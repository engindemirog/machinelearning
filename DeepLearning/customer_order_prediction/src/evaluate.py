from sklearn.metrics import classification_report

def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    y_pred_labels = (y_pred>0.5).astype("int32")

    print(classification_report(y_test,y_pred_labels))