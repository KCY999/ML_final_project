from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import data_process
from sklearn.model_selection import train_test_split
# 讀取數據
data = pd.read_csv("./processed_data/processed_train.csv")

# 提取目標變量和特徵
y = data['home_team_win']  # 提取目標變量
X = data.drop(columns=['home_team_win'])  # 剩餘的為特徵

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

max_num = 1500
tree_numbers = list(range(25, max_num+1, 25))
accuracy_scores = []

for n in tree_numbers:
    
    # 建立隨機森林模型
    rf = RandomForestClassifier(n_estimators=n, random_state=30)
    rf.fit(X_train, y_train)
    # 在測試集上評估準確率
    y_pred = rf.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# 畫圖觀察穩定性
plt.figure(figsize=(10, 6))
plt.plot(tree_numbers, accuracy_scores, marker='o')
plt.title('Effect of Tree Numbers on Model Stability', fontsize=14)
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True)
plt.savefig(f'./pic/tree_number_n{max_num}.png')
plt.show()


