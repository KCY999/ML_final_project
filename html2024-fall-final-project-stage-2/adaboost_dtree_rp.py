
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


best_adabD_model = AdaBoostClassifier(
    # random_state=42,
    estimator= DecisionTreeClassifier(max_depth=3),
    n_estimators=200,
    learning_rate=10,
)


best_adabD_model.fit(X, y)
y_pred = best_adabD_model.predict(X_test)

data_process.gen_submission_csv(y_pred)
