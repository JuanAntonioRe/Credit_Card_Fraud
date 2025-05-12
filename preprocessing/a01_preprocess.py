import pandas as pd

# Load of df
df = pd.read_csv('Dataset/raw/Credit_Card_Data.csv')

# Deleting the 'Unnamed: 0' column
df = df.drop('Unnamed: 0', axis=1)

# Showing values where 'activated_date' is NaN
print(df[df['activated_date'].isna()])

# Changing from objet to datetime 
df['activated_date'] = pd.to_datetime(df['activated_date'], format='%d/%m/%Y', errors='coerce')
df['last_payment_date'] = pd.to_datetime(df['last_payment_date'], format='%d/%m/%Y', errors='coerce')

# Changing the null values from activated_date
# I assume the df collection date is:
collection_date = pd.to_datetime('2020-11-09')

# Changing the null values
mask_activated_null = df['activated_date'].isna()
df.loc[mask_activated_null, 'activated_date'] = collection_date - pd.to_timedelta(df.loc[mask_activated_null, 'tenure'] * 30, unit='d')

print(df.info())
print()

# printing fraud value counts
print(df['fraud'].value_counts(normalize=True))

# Saving the new dataset
df.to_csv('Dataset/intermediate/preprocess_data.csv', index=False)