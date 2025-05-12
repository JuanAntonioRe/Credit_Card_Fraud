import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Verifying if the destination file exists
carpeta_detsino = "insight/"
if not os.path.exists(carpeta_detsino):
    # If the destination file does not exist, it is created
    os.makedirs(carpeta_detsino)


df = pd.read_csv('Dataset/intermediate/preprocess_data.csv')

# set the plot style
sns.set_style('whitegrid')

# plot the histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['balance'], bins=50, kde=True, color='blue')
plt.title('Hitogram of Balance Amount', fontsize=16)
plt.xlabel('Balance Amount ($)', fontsize=12)
plt.ylabel('Number of customers', fontsize=12)
plt.grid(False)
plt.savefig(f'{carpeta_detsino}/histogram_of_balance.png', dpi=300, bbox_inches='tight')
plt.show()

print('Figure have been saved')