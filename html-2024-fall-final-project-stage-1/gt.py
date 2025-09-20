import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score
import data_process
# 讀取數據
data = pd.read_csv("./processed_data/processed_train.csv")


y = data['home_team_win']
# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)


X_train = X_train.drop(columns=['home_pitcher', 'away_pitcher', 'home_team_win'])
X_test = X_test.drop(columns=['home_pitcher', 'away_pitcher', 'home_team_win'])

# # 帶投手數據處理
# p_win_rate = data_process.p_count(X_train)
# p_X_train = data_process.pitcher_win_proce(X_train, p_win_rate)
# p_X_test = data_process.pitcher_win_proce(X_test, p_win_rate)

# # 不帶投手數據處理
# np_X_train = X_train.drop(columns=['home_pitcher', 'away_pitcher', 'home_team_win'])
# np_X_test = X_test.drop(columns=['home_pitcher', 'away_pitcher', 'home_team_win'])

# p_X_train = p_X_train.drop(columns=['home_team_win'])
# p_X_test = p_X_test.drop(columns=['home_team_win'])

# 定義評估函數
def acc(y_hat, y):
    accuracy = np.mean(y == y_hat)
    return accuracy

# 定義參數網格
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 4, 5, 6,  7, 10],
    'min_samples_split': [5, 10, 20],
    'class_weight': ['balanced', None],
}



# 設定參數組合
grid = ParameterGrid(param_grid)
best_params = None
best_score = 0
results = []

# 網格搜索
for params in grid:
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # 訓練集準確率
    y_pred_train = model.predict(X_train)
    acc_train = acc(y_pred_train, y_train)
    
    # 測試集準確率
    y_pred_test = model.predict(X_test)
    acc_test = acc(y_pred_test, y_test)
    
    # 儲存結果
    results.append({
        "params": params,
        "train_accuracy": acc_train,
        "test_accuracy": acc_test
    })
    
    # 更新最佳參數
    if acc_test > best_score:
        best_score = acc_test
        best_params = params


print("Best Parameters:", best_params)
print("Best Test Accuracy:", best_score)


# 將結果整理為 DataFrame
results_df = pd.DataFrame(results)
results_df.sort_values(by="test_accuracy", ascending=False, inplace=True)

# 顯示最佳參數和結果
print("Best Parameters:", best_params)
print("Best Test Accuracy:", best_score)

results_df.to_csv("grid_search_results.csv", index=False)
print("Results saved to 'grid_search_results.csv'.")
