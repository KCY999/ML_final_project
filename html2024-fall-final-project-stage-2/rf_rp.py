from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import data_process

data = pd.read_csv("./processed_data/processed_train.csv")
y = data['home_team_win']
X = data.drop(columns=['home_team_win'])
X_test = pd.read_csv("./processed_data/processed_test.csv")


for i in range(5):
    best_rf_model = RandomForestClassifier(
        random_state=40+i,
        class_weight="balanced",
        max_depth=5,
        min_samples_split=20,
        n_estimators=200
    )

    best_rf_model.fit(X, y)
    y_pred = best_rf_model.predict(X_test)
    data_process.gen_submission_csv(y_pred, submit_name=f"rf_f1_submisson_{i+1}.csv")


for i in range(5):
    best_rf_model = RandomForestClassifier(
        random_state=40+i,
        class_weight=None,
        max_depth=9,
        min_samples_split=20,
        n_estimators=200
    )

    best_rf_model.fit(X, y)
    y_pred = best_rf_model.predict(X_test)

    data_process.gen_submission_csv(y_pred, submit_name=f"rf_acc_submisson_{i+1}.csv")
