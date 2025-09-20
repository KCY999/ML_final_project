import pandas as pd

df = pd.read_csv('__test.csv')

print(df.isnull().values.any())
missing_rows = df[df.isnull().any(axis=1)]
print("Rows with missing values:")
print(missing_rows)


