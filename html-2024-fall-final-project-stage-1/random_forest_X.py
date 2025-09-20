from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import data_process

data = pd.read_csv("./processed_data/processed_train.csv")


y = data['home_team_win']  # 目標變數
# X = data.drop(columns=['home_team_win'])  # 特徵變數



# y_test = pd.read_csv("./processed_data/processed_test.csv")




# 使用最佳參數訓練模型
best_rf_model = RandomForestClassifier(
    class_weight='balanced',
    max_depth=7,
    min_samples_split=10,
    n_estimators=200,
)

result = {
    "p_acc_in": 0,
    "p_acc_out": 0,
    "np_acc_in": 0,
    "np_acc_out": 0,
}


REPT = 10

def trainiing(X_train, X_test, y_train, y_test, picher=True):
    best_rf_model.fit(X_train, y_train)
    y_pred_in = best_rf_model.predict(X_train)
    y_pred_out = best_rf_model.predict(X_test)

    acc_in = accuracy_score(y_train, y_pred_in)
    print(f"ACC_in: {acc_in:.4f}")

    acc_out = accuracy_score(y_test, y_pred_out)
    print(f"ACC_val: {acc_out:.4f}")

    if picher:
        result["p_acc_in"] += acc_in/REPT
        result["p_acc_out"] += acc_out/REPT
    else:
        result["np_acc_in"] += acc_in/REPT
        result["np_acc_out"] += acc_out/REPT



for _ in range(REPT):
    
    # 分割數據集
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)



    p_win_rate = data_process.p_count(X_train)
    p_X_train = data_process.pitcher_win_proce(X_train, p_win_rate)
    p_X_test = data_process.pitcher_win_proce(X_test, p_win_rate)



    np_X_train = X_train.drop(columns=['home_pitcher','away_pitcher'])
    np_X_test = X_test.drop(columns=['home_pitcher','away_pitcher'])


    np_X_train = np_X_train.drop(columns=['home_team_win'])
    np_X_test = np_X_test.drop(columns=['home_team_win'])
    p_X_train = p_X_train.drop(columns=['home_team_win'])
    p_X_test = p_X_test.drop(columns=['home_team_win'])


    trainiing(p_X_train, p_X_test, y_train, y_test, picher=True)
    trainiing(np_X_train, np_X_test, y_train, y_test, picher=False)





print(result)


