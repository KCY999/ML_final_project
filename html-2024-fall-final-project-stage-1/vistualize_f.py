import matplotlib.pyplot as plt
import pandas as pd


# from scipy.stats import ttest_ind

# # for col in data.select_dtypes(include=['float64', 'int64']).columns:
# #     if col != "home_team_win":
# #         win_group = data[data['home_team_win'] == 1][col]
# #         lose_group = data[data['home_team_win'] == -1][col]
# #         t_stat, p_val = ttest_ind(win_group, lose_group)
# #         print(f"{col}: T-Stat = {t_stat:.4f}, P-Value = {p_val:.4f}")

# import seaborn as sns
# import matplotlib.pyplot as plt

# significant_features = [
#     "is_night_game",
#     "home_team_rest",
#     "away_team_rest",
#     "away_pitcher_rest",
#     "home_batting_batting_avg_10RA",
#     "home_batting_onbase_perc_10RA",
#     "home_batting_onbase_plus_slugging_10RA",
#     "home_batting_leverage_index_avg_10RA",
#     "home_batting_RBI_10RA",
#     "away_batting_onbase_perc_10RA",
#     "home_pitching_earned_run_avg_10RA",
#     # (繼續列舉P值小於0.05的特徵)
# ]

# columns_to_export = ["home_team_win"] + significant_features
# export_data = data[columns_to_export]

# # 匯出為 CSV 文件
# export_data.to_csv("home_team_win_with_significant_features.csv", index=False)



# # for col in significant_features:
# #     sns.boxplot(x='home_team_win', y=col, data=data)
# #     plt.title(f"Distribution of {col} by home_team_win")
# #     plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("./processed_data/processed_train.csv")
from scipy.stats import ttest_ind


# # 設置圖形大小
# plt.figure(figsize=(12, 6))

# # 對每個特徵繪製箱型圖
# for i, feature in enumerate(data.columns[:-1], 1):  # 忽略目標變數列
#     plt.subplot(1, len(data.columns[:-1]), i)
#     sns.boxplot(x='home_team_win', y=feature, data=data)
#     plt.title(f'{feature} vs home_team_win')
#     plt.xlabel('home_team_win')
#     plt.ylabel(feature)

# plt.tight_layout()
# plt.show()


# # 進行 t 檢驗
# for feature in data.columns[:-1]:  # 忽略目標變數列
#     group0 = data[data['home_team_win'] == 0][feature]
#     group1 = data[data['home_team_win'] == 1][feature]
#     t_stat, p_value = ttest_ind(group0, group1, equal_var=False)  # 假設方差不相等
#     print(f'{feature}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}')

# 分離目標變量和特徵
features = data.drop(columns=['home_team_win']).columns
target = 'home_team_win'
# 計算每個特徵在不同目標類別下的平均值
feature_means = data.groupby(target).mean()

# 繪製群組條形圖
x = np.arange(len(features))  # 每個特徵的位置
bar_width = 0.35             # 條形寬度

fig, ax = plt.subplots(figsize=(10, 6))

# 繪製條形圖
for i, class_label in enumerate(feature_means.index):
    ax.bar(
        x + i * bar_width,                    # 調整條形的位置
        feature_means.loc[class_label],       # 取出目標類別的特徵均值
        bar_width,                            # 條形寬度
        label=f"{target} = {class_label}"     # 標籤
    )

# 添加圖表標籤和格式化
ax.set_xlabel('Features')
ax.set_ylabel('Average Value')
ax.set_title(f'Features vs {target}')
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(features)
ax.legend()

# 顯示圖表
plt.tight_layout()
plt.show()