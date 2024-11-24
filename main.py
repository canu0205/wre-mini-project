import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# data preprocessing from 2001 to 2020
folder_path = './dataset'
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx')])
custom_columns = ['Date', 'Elevation(EL.m)', 'Storage Volume(million m^3)', 'Storage Rate(%)',
                  'Precipitation(mm)', 'Inflow(m^3/s)', 'Total Outflow(m^3/s)']
all_data = []

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    try:
        df = pd.read_excel(file_path, header=None, skiprows=3)
        df.columns = custom_columns
        # reverse the order of rows
        df = df.iloc[::-1].reset_index(drop=True)
        all_data.append(df)
    except Exception as e:
        print(f"Skipping file {file_name} due to error: {e}")

if all_data:
    combined_data = pd.concat(all_data, ignore_index=True)
    print(combined_data.head())
else:
    print("No valid data was found.")

# save combined data to excel
output_file_path = './output/combined_data.xlsx'
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
try:
    combined_data.to_excel(output_file_path, index=False,
                           sheet_name='Combined Data')
    print(f"Combined data successfully saved to {output_file_path}")
except Exception as e:
    print(f"Failed to save combined data due to error: {e}")

# data distribution
print(combined_data['Inflow(m^3/s)'].describe())
plt.figure(figsize=(12, 6))
sns.lineplot(data=combined_data, x='Date', y='Inflow(m^3/s)')
plt.title('Daily Inflow Over Time')
plt.show()

sns.histplot(combined_data['Inflow(m^3/s)'], kde=True, bins=30)
plt.title('Inflow Distribution')
plt.show()

# statistical analysis
