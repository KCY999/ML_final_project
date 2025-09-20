from sklearn.linear_model import Lasso
import pandas as pd

data = pd.read_csv("./processed_data/processed_train.csv")
y = data['home_team_win']
X = data.drop(columns=['home_team_win'])

# 使用 Lasso 進行特徵選擇
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
lasso_features = X.columns[lasso.coef_ != 0].tolist()
print("Lasso 選擇的特徵:", lasso_features)
