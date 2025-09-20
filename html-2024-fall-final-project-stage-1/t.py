
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

data = pd.read_csv("./processed_data/processed_train.csv")
y = data['home_team_win']
X = data.drop(columns=['home_team_win'])


# 定義基礎分類器
base_estimator = DecisionTreeClassifier()

# 定義 Adaboost 模型
adaboost = AdaBoostClassifier(estimator=base_estimator, random_state=42)

# 定義網格搜索的參數範圍
param_grid = {
    'n_estimators': [50, 100, 200],  # 弱分類器數量
    'learning_rate': [0.01, 0.1, 1, 10],  # 學習率
    'estimator__max_depth': [1, 2, 3, 4]  # 基礎分類器（決策樹）的最大深度
}

# 設置 GridSearchCV
grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, 
                           scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# 執行網格搜索
grid_search.fit(X, y)

# 提取結果
results = grid_search.cv_results_
mean_test_scores = results['mean_test_score']
param_combinations = results['params']

# 將結果轉換為可視化數據
n_estimators = sorted(list(set([param['n_estimators'] for param in param_combinations])))
learning_rates = sorted(list(set([param['learning_rate'] for param in param_combinations])))

# 繪製網格搜索結果
plt.figure(figsize=(12, 8))
for depth in sorted(list(set([param['estimator__max_depth'] for param in param_combinations]))):
    for lr in learning_rates:
        scores = []
        for n in n_estimators:
            score = [mean_test_scores[i] for i, param in enumerate(param_combinations)
                     if param['learning_rate'] == lr and param['n_estimators'] == n and param['estimator__max_depth'] == depth]
            scores.append(score[0])
        plt.plot(n_estimators, scores, marker='o', label=f'Depth={depth}, LR={lr}')

plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Mean CV Accuracy')
plt.title('Grid Search Results for AdaBoost (Combined Plot)')
plt.legend(title='Params (Depth, Learning Rate)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


plt.savefig("./pic/adaboost_grid_results_lineplot_.png")
plt.show()

