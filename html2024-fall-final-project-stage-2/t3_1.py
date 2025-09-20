import numpy as np
import pandas as pd
import s2_process
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# 讀取數據
data = pd.read_csv("./processed_data/processed_train.csv")

# y, X are DataFrame
y = data['home_team_win']
X = data[['pitcher_rest_diff', 'batting_onbase_plus_slugging_10RA_diff', 'batting_leverage_index_avg_10RA_diff', 'pitching_earned_run_avg_10RA_diff', 'pitching_SO_batters_faced_10RA_diff', 'pitching_H_batters_faced_10RA_diff', 'pitching_BB_batters_faced_10RA_diff', 'team_errors_mean_diff', 'team_errors_skew_diff', 'team_wins_skew_diff', 'batting_onbase_perc_mean_diff', 'batting_onbase_plus_slugging_mean_diff', 'batting_wpa_bat_mean_diff', 'batting_wpa_bat_skew_diff', 'batting_RBI_std_diff', 'batting_RBI_skew_diff', 'pitching_earned_run_avg_std_diff', 'pitching_SO_batters_faced_mean_diff', 'pitching_SO_batters_faced_std_diff', 'pitching_H_batters_faced_mean_diff', 'pitching_H_batters_faced_std_diff', 'pitching_H_batters_faced_skew_diff', 'pitching_BB_batters_faced_std_diff', 'pitching_leverage_index_avg_std_diff', 'pitching_leverage_index_avg_skew_diff', 'pitching_wpa_def_mean_diff', 'pitching_wpa_def_std_diff', 'pitching_wpa_def_skew_diff', 'pitcher_earned_run_avg_mean_diff', 'pitcher_SO_batters_faced_mean_diff', 'pitcher_SO_batters_faced_std_diff', 'pitcher_SO_batters_faced_skew_diff', 'pitcher_H_batters_faced_std_diff', 'pitcher_H_batters_faced_skew_diff', 'pitcher_BB_batters_faced_mean_diff', 'pitcher_BB_batters_faced_skew_diff', 'pitcher_leverage_index_avg_mean_diff', 'pitcher_wpa_def_mean_diff', 'pitcher_wpa_def_std_diff', 'pitcher_wpa_def_skew_diff']]
X = data.drop(columns=['home_team_win'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)




def acc(y_hat, y):
    accuracy = 0
    for i in range(len(y)):
        if y.iloc[i] == np.sign(y_hat[i]):
            accuracy += 1
    # print("0/1  acc:", accuracy/len(y))
    return accuracy/len(y)

# 初始化模型
models = {
    "LinearRegression": LinearRegression(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(max_depth=7, n_estimators=200),
    # "RandomForestClassifier": RandomForestClassifier(    class_weight='balanced',
    # max_depth=10,
    # min_samples_split=10,
    # n_estimators=200,),
    # "GradientBoostingClassifier": GradientBoostingClassifier()
}

REPT = 1
acc_dict = { model_name:0 for model_name in models}
for _ in range(REPT):
    # 對每個模型進行訓練和測試
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        # 訓練模型
        model.fit(X_train, y_train)
        y_pred_in = model.predict(X_train)
        acc_in = acc(y_pred_in, y_train)
        # acc_in = accuracy_score(y_train, y_pred_in)
        print(f"ACC_in: {acc_in:.4f}")
        
        # 預測
        y_pred_out = model.predict(X_test)
        print(y_pred_out)
        # print(type(y_pred_out))
        # 直接使用 sklearn 的 accuracy_score
        acc_out = acc(y_pred_out, y_test)
        print(f"ACC_val: {acc_out:.4f}")
        acc_dict[model_name] += acc_out


for model_name, acc_val in acc_dict.items():
    print(model_name, ":", acc_val/REPT)