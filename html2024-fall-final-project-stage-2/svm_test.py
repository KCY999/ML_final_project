from sklearn.svm import SVC  # SVM for classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("./processed_data/processed_train.csv")

# 假設 `data` 是已處理完成的 DataFrame
y = data['home_team_win']  # 目標變數
X = data.drop(columns=['home_team_win'])  # 特徵變數

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 'scale'],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)

# 最佳參數
print("Best Parameters:", grid_search.best_params_)

# 使用最佳參數進行測試
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))