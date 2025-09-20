from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import data_process
import json

# 讀取數據
data = pd.read_csv("./processed_data/processed_train.csv")
y = data['home_team_win']  # 目標變數

# 結果保存字典
result = {
    "p_acc_out": [],
    "np_acc_out": [],
}

# 網格搜索參數
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [3, 4, 5, 6, 7, 8, 10, 20],
    'min_samples_split': [5, 10, 20],
    'class_weight': ['balanced', None],
    # 'max_features': ['sqrt', 'log2'],
    # 'min_samples_leaf': [1, 2, 5]
}


def train_and_evaluate(X_train, X_test, y_train, y_test, param_grid, picher=True):
    best_acc_out = 0
    best_params = None

    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for class_weight in param_grid['class_weight']:
                    # for max_features in param_grid['max_features']:
                        # for min_samples_leaf in param_grid['min_samples_leaf']:
                            # 初始化模型
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                class_weight=class_weight,
                                # max_features=max_features,
                                # min_samples_leaf=min_samples_leaf,
                            )
                            # 訓練模型
                            model.fit(X_train, y_train)
                            # 預測測試集
                            y_pred_out = model.predict(X_test)
                            acc_out = accuracy_score(y_test, y_pred_out)

                            # 更新最佳參數
                            if acc_out > best_acc_out:
                                best_acc_out = acc_out
                                best_params = {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'class_weight': class_weight,
                                    # 'max_features': max_features,
                                    # 'min_samples_leaf': min_samples_leaf
                                }

    print(("picher" if picher else "non-picher") + " :")
    print(f"Best Params: {best_params}")
    print(f"Best ACC_out: {best_acc_out:.4f}")

    # 累加結果
    if picher:
        result["p_acc_out"].append({
            "best_params": best_params,
            "best_acc_out": best_acc_out
        })
    else:
        result["np_acc_out"].append({
            "best_params": best_params,
            "best_acc_out": best_acc_out
        })


# 重複實驗次數
REPT = 5  # 可以更改為任意次數

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

    # 訓練與評估
    train_and_evaluate(p_X_train, p_X_test, y_train, y_test, param_grid, picher=True)
    train_and_evaluate(np_X_train, np_X_test, y_train, y_test, param_grid, picher=False)


with open(f"grid_search_AO_R{REPT}.json", "w") as f:
    json.dump(result, f, indent=4)

print("complete")