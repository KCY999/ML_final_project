import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data = pd.read_csv("./processed_data/processed_train.csv")
y = data['home_team_win']
X = data.drop(columns=['home_team_win'])

# # 切分訓練集和測試集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定義參數網格
param_grid = {
    'max_depth': [5, 7, 9],
    'min_samples_split': [5, 10, 20],
    'class_weight': [None, 'balanced']
}

# 建立隨機森林模型
rf = RandomForestClassifier(random_state=42, n_estimators=200) 

# 使用 GridSearchCV 進行參數搜索
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1_macro', verbose=1, n_jobs=-1)
grid_search.fit(X, y)

# 提取最佳參數和分數
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_score)

# 提取 GridSearchCV 結果
results = pd.DataFrame(grid_search.cv_results_)

# 繪製所有參數對準確率的綜合影響（折線圖）
plt.figure(figsize=(12, 8))
markers = ['o', 's', '^', 'D']

for i, depth in enumerate(param_grid['max_depth']):
    for j, split in enumerate(param_grid['min_samples_split']):
        subset = results[(results['param_max_depth'] == depth) & (results['param_min_samples_split'] == split)]
        class_weights = subset['param_class_weight'].astype(str)
        accuracies = subset['mean_test_score']

        plt.plot(
            class_weights, 
            accuracies, 
            marker=markers[j % len(markers)], 
            label=f'max_depth={depth}, min_samples_split={split}'
        )

# 設置圖表
plt.title('Effect of Grid Parameters on Accuracy', fontsize=16)
plt.xlabel('Class Weight', fontsize=14)
plt.ylabel('Mean Cross-Validation Accuracy', fontsize=14)
plt.legend(title='Parameter Combinations', fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./pic/grid_search_combined_lineplot_f1.png')
plt.show()
