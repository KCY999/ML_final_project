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
X = data.drop(columns=['home_team_win'])  # 特徵變數

# y_test = pd.read_csv("./processed_data/processed_test.csv")




# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'class_weight': ['balanced', None]
# }

# grid_search = GridSearchCV(
#     RandomForestClassifier(random_state=42),
#     param_grid=param_grid,
#     cv=5,
#     scoring='f1_macro'
# )
# grid_search.fit(X_train, y_train)
# # Best Parameters: {'class_weight': 'balanced', 'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 200}
# print("Best Parameters:", grid_search.best_params_)



# 使用最佳參數訓練模型
best_rf_model = RandomForestClassifier(
    class_weight='balanced',
    max_depth=15,
    min_samples_split=10,
    n_estimators=200,
)


best_rf_model.fit(X, y)
y_pred = best_rf_model.predict(X_test)

# data_process.gen_submission_csv(y_pred)

# 評估結果
print(classification_report(y_test, y_pred))





# # 繪製混淆矩陣
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf_model.classes_)
# disp.plot()
# plt.show()
# feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
# print(feature_importances.head(10))  # 輸出前 10 個重要特徵

# # 繪製特徵重要性
# plt.figure(figsize=(10, 6))
# feature_importances.plot(kind='bar')
# plt.title("Top Feature Importances")
# plt.show()

