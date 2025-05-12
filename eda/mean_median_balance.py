import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Verifying if the destination file exists
carpeta_detsino = "insight/"
if not os.path.exists(carpeta_detsino):
    # If the destination file does not exist, it is created
    os.makedirs(carpeta_detsino)
    

df = pd.read_csv('Dataset/intermediate/preprocess_data.csv', parse_dates=['activated_date'])

# Extract year an month from 'activated_date'
df['activation_year'] = df['activated_date'].dt.year
df['activation_month'] = df['activated_date'].dt.month

# Group by year and month
balance_stats = df.groupby(['activation_year', 'activation_month'])['balance'].agg(['mean', 'median']).reset_index()

# Columns rename
balance_stats.columns = ['Year', 'Month', 'Mean_Balance', 'Median_Balance']

# New column of fatetime type to represent the first day of the month. This helps to plot 2 years in one line
balance_stats['activation_period'] = pd.to_datetime(balance_stats[['Year', 'Month']].assign(DAY=1))

# Sort by the new column
balance_stats = balance_stats.sort_values('activation_period')

plt.figure(figsize=(12, 6))

# Mean Blance plot
sns.lineplot(
    data=balance_stats,
    x='activation_period',
    y='Mean_Balance',
    marker='o',
    label='Mean Balance'
)

# Median Balance plot
sns.lineplot(
    data=balance_stats,
    x='activation_period',
    y='Median_Balance',
    marker='o',
    label='Median Balance'
)

# Adding notes in the points
for i in range(len(balance_stats)):
    x = balance_stats['activation_period'].iloc[i]
    mean_y = balance_stats['Mean_Balance'].iloc[i]
    median_y = balance_stats['Median_Balance'].iloc[i]
    
    # Notes
    plt.text(x, mean_y + 35, f"{mean_y:.0f}", ha='center', fontsize=8, color='blue')
    plt.text(x, median_y + 35, f"{median_y:.0f}", ha='center', fontsize=8, color='orange')

plt.title("Mean and Median Balance")
plt.xlabel("Activation Date (Year-Month)")
plt.ylabel("Balance Amount ($)")
plt.xticks(rotation=45)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig(f'{carpeta_detsino}/mean_madian_balance.png', dpi=300, bbox_inches='tight')
plt.show()

print('Figure have been saved')