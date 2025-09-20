from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import data_process
import json

# 讀取數據
data = pd.read_csv("./processed_data/processed_train.csv")
y = data['home_team_win']  # 目標變數

# 結果保存字典
result = {
    "p_grid_search": [],
    "np_grid_search": [],
}

# 網格搜索參數
param_grid = {
    # 'n_estimators': [200, 300, 400],
    # 'max_depth': [3, 4, 5, 6,  7, 10],
    # 'min_samples_split': [5, 10, 20],
    # 'class_weight': ['balanced', None],
    

    'n_estimators': [200, 300],
    'max_depth': [3, 4, 5, 6,  7, 8, 10, 20],
    'min_samples_split': [5, 10, 20],
    'class_weight': ['balanced', None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 5]
}



def train_and_evaluate(X_train, X_test, y_train, y_test, grid_search, picher=True):
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred_in = best_model.predict(X_train)
    y_pred_out = best_model.predict(X_test)

    acc_in = accuracy_score(y_train, y_pred_in)
    acc_out = accuracy_score(y_test, y_pred_out)


    print(("picher" if picher else "non-picher") + " :")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"ACC_in: {acc_in:.4f}, ACC_out: {acc_out:.4f}")

    # 累加結果
    if picher:
        result["p_grid_search"].append({
            "best_params": grid_search.best_params_,
            "acc_in": acc_in,
            "acc_out": acc_out
        })
    else:
        result["np_grid_search"].append({
            "best_params": grid_search.best_params_,
            "acc_in": acc_in,
            "acc_out": acc_out
        })

# 重複實驗次數
REPT =  1  # 可以更改為任意次數

for _ in range(REPT):
    # 分割數據集
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)

    # 帶投手數據處理
    p_win_rate = data_process.p_count(X_train)
    p_X_train = data_process.pitcher_win_proce(X_train, p_win_rate)
    p_X_test = data_process.pitcher_win_proce(X_test, p_win_rate)

    # 不帶投手數據處理
    np_X_train = X_train.drop(columns=['home_pitcher', 'away_pitcher', 'home_team_win'])
    np_X_test = X_test.drop(columns=['home_pitcher', 'away_pitcher', 'home_team_win'])

    p_X_train = p_X_train.drop(columns=['home_team_win'])
    p_X_test = p_X_test.drop(columns=['home_team_win'])

    # 定義GridSearchCV
    grid_search_p = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search_np = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=-1)

    # 訓練與評估
    train_and_evaluate(p_X_train, p_X_test, y_train, y_test, grid_search_p, picher=True)
    train_and_evaluate(np_X_train, np_X_test, y_train, y_test, grid_search_np, picher=False)


with open(f"grid_search_results_R{REPT}.json", "w") as f:
    json.dump(result, f, indent=4)

print("結果保存至 grid_search_results.json")