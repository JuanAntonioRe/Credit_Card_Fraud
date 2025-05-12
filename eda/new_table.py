import pandas as pd

df = pd.read_csv('Dataset/intermediate/preprocess_data.csv', parse_dates=['activated_date','last_payment_date'])

# Filter by activated date and last payment in 2020
df_2020 = df[
    (df['activated_date'].dt.year == 2020) &
    (df['last_payment_date'].dt.year == 2020)
].copy()

# Just numbers in the 'cust_id'
df_2020['cust_id_clean'] = df_2020['cust_id'].astype(str).str.extract(r'(\d+)')

# New column with the % of 'cash_advance' on 'credit_limit'
df_2020['%_cash_advance'] = (df_2020['cash_advance'] / df_2020['credit_limit']) * 100

# Changing format for 'activated_date'
df_2020['activated_date'] = df['activated_date'].dt.to_period('M')

# Required columns
result_table = df_2020[[
    'cust_id_clean',
    'activated_date',
    'last_payment_date',
    'cash_advance',
    'credit_limit',
    '%_cash_advance'
]].copy()

print(result_table.info())
print(result_table.head(3))

# Saving the new dataset .csv and .xlsx file
df.to_csv('Dataset/output/new_table.csv', index=False)
df.to_excel('Dataset/output/newtable.xlsx', sheet_name='Report', index=False, engine='openpyxl')

print('\nNew table saved')