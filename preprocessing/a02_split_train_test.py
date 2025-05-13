import pandas as pd

# dataset load
df = pd.read_csv('Dataset/intermediate/preprocess_data.csv')

# Deleting Null values
df = df.dropna()

# Top 10 features
top_10_features = [
    'payments', 'purchases', 'credit_limit', 'minimum_payments', 'oneoff_purchases',
    'balance', 'cash_advance', 'cash_advance_trx', 'purchases_trx', 'installments_purchases'
]

# features and target
X = df[top_10_features]
y = df['fraud']

X.to_csv("Dataset/intermediate/X.csv", index=False)
y.to_csv("Dataset/intermediate/y.csv", index=False)

print('\nFeatures and target saved')