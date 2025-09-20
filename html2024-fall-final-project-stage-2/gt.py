import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score

# 讀取數據
data = pd.read_csv("./processed_data/processed_train.csv")

# y, X are DataFrame
y = data['home_team_win']
X = data.drop(columns=['home_team_win'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 定義評估函數
def acc(y_hat, y):
    accuracy = np.mean(y == y_hat)
    return accuracy

# 定義參數網格
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 12, 13, 14, 15, 17, 18, 19, 20, None],
    "min_samples_split": [2, 10],
    "class_weight": [None, "balanced"]
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

# 將結果整理為 DataFrame
results_df = pd.DataFrame(results)
results_df.sort_values(by="test_accuracy", ascending=False, inplace=True)

# 顯示最佳參數和結果
print("Best Parameters:", best_params)
print("Best Test Accuracy:", best_score)

results_df.to_csv("grid_search_results.csv", index=False)
print("Results saved to 'grid_search_results.csv'.")
