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


# models = {
#     "LogisticRegression": LogisticRegression(C=1.0, max_iter=500),  # 增大 C，允許更多自由度
#     "RandomForestClassifier": RandomForestClassifier(
#         max_depth=10, min_samples_leaf=5, random_state=42  # 避免過擬合
#     ),
#     "GradientBoostingClassifier": GradientBoostingClassifier(
#         learning_rate=0.05, n_estimators=300, max_depth=3, random_state=42
#     )  # 調整學習率和樹的深度
# }

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(C=5.0, max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, max_depth=3, random_state=42)
}

# 訓練與評估
for model_name, model in models.items():
    print(f"\n--- {model_name} ---")
    
    # Logistic Regression 使用多項式特徵
    if model_name == "LogisticRegression":
        model.fit(X_train_poly, y_train)
        y_pred_in = model.predict(X_train_poly)
        y_pred_out = model.predict(X_test_poly)
    else:
        model.fit(X_train, y_train)
        y_pred_in = model.predict(X_train)
        y_pred_out = model.predict(X_test)
    
    # 訓練準確度
    acc_in = accuracy_score(y_train, y_pred_in)
    print(f"ACC_in: {acc_in:.4f}")
    
    # 驗證準確度
    acc_out = accuracy_score(y_test, y_pred_out)
    print(f"ACC_val: {acc_out:.4f}")
