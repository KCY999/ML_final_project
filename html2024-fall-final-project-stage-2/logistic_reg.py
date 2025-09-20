from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import s2_process
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("./processed_data/processed_train.csv")
# data = pd.read_csv("./home_team_win_with_significant_features.csv")

# y, X are DataFrame
y, X = s2_process.get_y_and_x(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=1, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


model = LogisticRegression(C=1, max_iter=1000)
model.fit(X_train_poly, y_train)


y_pred_in = model.predict(X_train_poly)
y_pred_out = model.predict(X_test_poly)

acc_in = accuracy_score(y_train, y_pred_in)
print(f"ACC_in: {acc_in:.4f}")

acc_out = accuracy_score(y_test, y_pred_out)
print(f"ACC_val: {acc_out:.4f}")

from sklearn.model_selection import cross_val_score

# scores = cross_val_score(model, X_train_poly, y_train, cv=5)
# print(f"Cross-validated ACC: {scores.mean():.4f}")