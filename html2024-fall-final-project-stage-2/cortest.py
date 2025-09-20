import pandas as pd
from scipy.stats import chi2_contingency
from matplotlib import pyplot as plt
import seaborn as sns
data = pd.read_csv("./processed_data/processed_train.csv")


# 假設 data 是你的 DataFrame
features = data.drop(columns=['home_team_win']).columns
target = 'home_team_win'

# 計算相關係數
correlation = data[features].corrwith(data[target])

# 排序
correlation_sorted = correlation.sort_values(ascending=False)

# 顯示結果
print(correlation_sorted)
# 整理相關係數為 DataFrame
correlation_df = correlation_sorted.reset_index()
correlation_df.columns = ['Feature', 'Correlation']

# 繪製條形圖
plt.figure(figsize=(10, 6))
sns.barplot(x='Correlation', y='Feature', data=correlation_df, palette='viridis')
plt.title('Correlation with Target: home_team_win')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.tight_layout()
plt.show()



# X = data.drop(columns=['home_team_win'])
# y = data['home_team_win']



# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier()
# model.fit(X, y)

# # 提取特徵重要性
# importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
# print(importances)

