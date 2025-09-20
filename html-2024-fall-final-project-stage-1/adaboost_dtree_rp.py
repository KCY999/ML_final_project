
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import data_process

data = pd.read_csv("./processed_data/processed_train.csv")
y = data['home_team_win']
X = data.drop(columns=['home_team_win'])
X_test = pd.read_csv("./processed_data/processed_test.csv")


# for i in range(5):
#     best_adabD_model = AdaBoostClassifier(
#         random_state=40+i,
#         estimator= DecisionTreeClassifier(max_depth=3, random_state=40+i),
#         n_estimators=200,
#         learning_rate=10,
#     )

#     best_adabD_model.fit(X, y)
#     y_pred = best_adabD_model.predict(X_test)

#     data_process.gen_submission_csv(y_pred, submit_name=f"adabDt_f1_submisson_{i+1}.csv")
    
for i in range(5):
    best_adabD_model = AdaBoostClassifier(
        random_state=40+i,
        estimator= DecisionTreeClassifier(max_depth=1, random_state=40+i),
        n_estimators=50,
        learning_rate=0.1,
    )

    best_adabD_model.fit(X, y)
    y_pred = best_adabD_model.predict(X_test)

    data_process.gen_submission_csv(y_pred, submit_name=f"adabDt_acc_submisson_{i+1}.csv")
