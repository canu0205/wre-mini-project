import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

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
        all_data.append(df)
    except Exception as e:
        print(f"Skipping file {file_name} due to error: {e}")

if all_data:
    combined_data = pd.concat(all_data, ignore_index=True)
    print(combined_data.head())
else:
    print("No valid data was found.")
