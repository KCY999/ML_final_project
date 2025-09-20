from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import data_process

data = pd.read_csv("./processed_data/processed_train.csv")

# 假設 data 是你的數據集
y = data['home_team_win']  # 目標變數
X_test = pd.read_csv("./processed_data/processed_test.csv")

# X = data.drop(columns=['home_team_win'])  # 特徵變數
# X = X.drop(columns=['home_pitcher','away_pitcher'])
# X_test = X_test.drop(columns=['home_pitcher','away_pitcher'])

X = data
p_win_rate = data_process.p_count(X)
X = data.drop(columns=['home_team_win'])
X = data_process.pitcher_win_proce(X, p_win_rate)
X_test = data_process.pitcher_win_proce(X_test, p_win_rate)

best_rf_model = RandomForestClassifier(
    class_weight=None,
    max_depth=4,
    min_samples_split=10,
    n_estimators=300
)


best_rf_model.fit(X, y)
y_pred = best_rf_model.predict(X_test)

data_process.gen_submission_csv(y_pred)


