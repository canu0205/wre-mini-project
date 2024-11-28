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
# output_file_path = './output/combined_data.xlsx'
# os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
# try:
#     combined_data.to_excel(output_file_path, index=False,
#                            sheet_name='Combined Data')
#     print(f"Combined data successfully saved to {output_file_path}")
# except Exception as e:
#     print(f"Failed to save combined data due to error: {e}")

# Frequency analysis
inflow = combined_data['Inflow(m^3/s)'].dropna()

# Remove outliers using IQR method
Q1 = inflow.quantile(0.25)
Q3 = inflow.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
inflow_no_outliers = inflow[(inflow >= lower_bound) & (inflow <= upper_bound)]

print(f"Original data size: {len(inflow)}")
print(f"Data size after outlier removal: {len(inflow_no_outliers)}")

# Visualize data before and after outlier removal
# Histogram before outlier removal
plt.figure(figsize=(10, 6))
sns.histplot(inflow, bins=50, kde=True, color='skyblue')
plt.title('Histogram of Inflow Data (Before Outlier Removal)')
plt.xlabel('Inflow (m³/s)')
plt.ylabel('Frequency')
plt.show()

# Histogram after outlier removal
plt.figure(figsize=(10, 6))
sns.histplot(inflow_no_outliers, bins=50, kde=True, color='green')
plt.title('Histogram of Inflow Data (After Outlier Removal)')
plt.xlabel('Inflow (m³/s)')
plt.ylabel('Frequency')
plt.show()

# Proceed with positive inflow values
positive_inflow = inflow_no_outliers[inflow_no_outliers > 0]

distributions = {
    'Normal': stats.norm,
    'Log-Normal': stats.lognorm,
    'Gamma': stats.gamma,
    'Log-Pearson Type III': stats.pearson3,
    'Gumbel': stats.gumbel_r,
    'Weibull': stats.weibull_min
}

# plt.figure(figsize=(12, 6))
# sns.histplot(inflow, kde=True, bins=700, label='Inflow Data')
# plt.title('Histogram of Inflow Data')
# plt.xlabel('Inflow (m³/s)')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()


# Function to combine bins with expected counts less than 5
def combine_bins(observed_counts, expected_counts, bin_edges):
    """
    Combines bins with expected counts less than 5.
    """
    i = 0
    while i < len(expected_counts):
        if expected_counts[i] < 5:
            if i == len(expected_counts) - 1:
                # Merge with the previous bin
                expected_counts[i - 1] += expected_counts[i]
                observed_counts[i - 1] += observed_counts[i]
                expected_counts = np.delete(expected_counts, i)
                observed_counts = np.delete(observed_counts, i)
                bin_edges = np.delete(bin_edges, i + 1)
                i -= 1
            else:
                # Merge with the next bin
                expected_counts[i + 1] += expected_counts[i]
                observed_counts[i + 1] += observed_counts[i]
                expected_counts = np.delete(expected_counts, i)
                observed_counts = np.delete(observed_counts, i)
                bin_edges = np.delete(bin_edges, i + 1)
        else:
            i += 1
    return observed_counts, expected_counts, bin_edges


# Fit distributions
fitted_params = {}
for name, dist in distributions.items():
    try:
        if name in ['Log-Normal', 'Gamma', 'Weibull', 'Log-Pearson Type III']:
            data = positive_inflow
            if name == 'Log-Normal':
                params = dist.fit(data, floc=0)
            else:
                params = dist.fit(data)
        else:
            data = inflow
            params = dist.fit(data)
        fitted_params[name] = params
        print(f"Fitted parameters for {name}: {params}")
    except Exception as e:
        print(f"Error fitting {name}: {e}")

# Initialize test results
chi_square_results = {}
ks_test_results = {}

for name, dist in distributions.items():
    params = fitted_params.get(name)
    if params:
        if name in ['Log-Normal', 'Gamma', 'Weibull', 'Log-Pearson Type III']:
            data = positive_inflow
        else:
            data = inflow

        # Adjust bin count for each distribution if necessary
        bin_count = 700

        # Generate observed frequencies
        observed_counts, bin_edges = np.histogram(data, bins=bin_count)

        # Compute expected frequencies by integrating the PDF over the bins
        expected_counts = []
        for i in range(len(bin_edges) - 1):
            cdf_upper = dist.cdf(bin_edges[i + 1], *params)
            cdf_lower = dist.cdf(bin_edges[i], *params)
            prob = cdf_upper - cdf_lower
            expected_count = prob * len(data)
            expected_counts.append(expected_count)

        expected_counts = np.array(expected_counts)
        observed_counts = np.array(observed_counts)

        # Adjust expected counts to match the sum of observed counts
        total_expected = np.sum(expected_counts)
        total_observed = np.sum(observed_counts)
        if total_expected > 0:
            expected_counts *= total_observed / total_expected
        else:
            print(f"Total expected counts is zero for {
                  name}. Skipping this distribution.")
            continue

        # Combine bins with expected counts less than 5
        observed_counts, expected_counts, bin_edges = combine_bins(
            observed_counts, expected_counts, bin_edges)

        # Ensure there are enough bins after combining
        if len(observed_counts) < 2:
            print(f"Not enough bins for {
                  name} after combining. Skipping this distribution.")
            continue

        # Re-adjust expected counts to match observed counts sum
        total_expected = np.sum(expected_counts)
        total_observed = np.sum(observed_counts)
        if total_expected > 0:
            expected_counts *= total_observed / total_expected

        # Perform Chi-square test
        try:
            chi_square_stat, chi_square_p = stats.chisquare(
                f_obs=observed_counts, f_exp=expected_counts)
            chi_square_results[name] = {
                'Chi-Square Statistic': chi_square_stat, 'p-value': chi_square_p}
        except Exception as e:
            print(f"Error in Chi-square test for {name}: {e}")
            continue

        # Perform KS test
        ks_stat, ks_p = stats.kstest(data, dist.cdf, args=params)
        ks_test_results[name] = {'KS Statistic': ks_stat, 'p-value': ks_p}

# Compare distributions using test statistics
# Rank distributions by Chi-square Statistic (lower is better)
chi_square_ranking = sorted(
    chi_square_results.items(), key=lambda x: x[1]['Chi-Square Statistic'])
# Rank distributions by KS Statistic (lower is better)
ks_ranking = sorted(ks_test_results.items(),
                    key=lambda x: x[1]['KS Statistic'])

print("\nChi-square Test Results (Ranked by Chi-square Statistic):")
for name, res in chi_square_ranking:
    print(f"{name}: Chi-Square Statistic = {
          res['Chi-Square Statistic']:.4f}, p-value = {res['p-value']:.4e}")

print("\nKolmogorov-Smirnov Test Results (Ranked by KS Statistic):")
for name, res in ks_ranking:
    print(f"{name}: KS Statistic = {
          res['KS Statistic']:.4f}, p-value = {res['p-value']:.4e}")

# Choose the best-fit distribution based on minimum Chi-square Statistic
best_fit_name_chi = chi_square_ranking[0][0]
# Choose the best-fit distribution based on minimum KS Statistic
best_fit_name_ks = ks_ranking[0][0]

print(
    f"\nBest-fit distribution based on Chi-square Statistic: {best_fit_name_chi}")
print(f"Best-fit distribution based on KS Statistic: {best_fit_name_ks}")

# Calculate return period inflows for the best-fit distribution (Chi-square)
if best_fit_name_chi in fitted_params:
    best_fit_params = fitted_params[best_fit_name_chi]
    T = [50, 100, 500]
    try:
        return_period_inflows = [distributions[best_fit_name_chi].ppf(
            1 - 1/t, *best_fit_params) for t in T]
        # Format the results to four decimal places
        formatted_inflows = [
            f"{inflow_value:.4f}" for inflow_value in return_period_inflows]
        print(f"\nReturn Period Inflows ({best_fit_name_chi}): {
              formatted_inflows}")
    except Exception as e:
        print(f"Error calculating return periods for {best_fit_name_chi}: {e}")
else:
    print(f"No fitted parameters found for {
          best_fit_name_chi}. Cannot calculate return period inflows.")

# New data for 2023
new_data_path = os.path.join('./dataset', os.listdir('./dataset')[0])
new_data = pd.read_excel(new_data_path, header=None, skiprows=3)
new_data.columns = custom_columns
new_inflow = new_data['Inflow(m^3/s)'].dropna()

# Calculate the 25th, 50th, and 75th percentile inflow of the new data
q1 = np.percentile(new_inflow, 25)
q2 = np.percentile(new_inflow, 50)
q3 = np.percentile(new_inflow, 75)
print(f"\n2023 Data Percentiles:")
print(f"25th Percentile (Q1): {q1:.4f}")
print(f"50th Percentile (Q2): {q2:.4f}")
print(f"75th Percentile (Q3): {q3:.4f}")

# Remove outliers from the new data using IQR method
iqr_new = q3 - q1
lower_bound_new = q1 - 1.5 * iqr_new
upper_bound_new = q3 + 1.5 * iqr_new
new_inflow_filtered = new_inflow[(new_inflow >= lower_bound_new) & (
    new_inflow <= upper_bound_new)]

print(f"Original new data size: {len(new_inflow)}")
print(f"Filtered new data size: {len(new_inflow_filtered)}")

# Estimate return period of each inflow percentile using the best-fit model from Chi-square test
best_fit_params = fitted_params.get(best_fit_name_chi)
if best_fit_params:
    # For distributions that require positive data
    if best_fit_name_chi in ['Log-Normal', 'Gamma', 'Weibull', 'Log-Pearson Type III']:
        percentiles = [q1, q2, q3]
        percentiles = [max(p, 0.0001) for p in percentiles]
        exceedance_probs = [
            1 - distributions[best_fit_name_chi].cdf(p, *best_fit_params) for p in percentiles]
    else:
        percentiles = [q1, q2, q3]
        exceedance_probs = [
            1 - distributions[best_fit_name_chi].cdf(p, *best_fit_params) for p in percentiles]

    # Estimate return periods
    estimated_return_periods = [1 / prob if prob >
                                0 else np.inf for prob in exceedance_probs]

    # Format and display the results
    print(f"\nEstimated Return Periods using {
          best_fit_name_chi} distribution:")
    for i, percentile in enumerate(['25th', '50th', '75th']):
        return_period = estimated_return_periods[i]
        if np.isinf(return_period):
            print(f"{percentile} Percentile Inflow: Return Period is greater than available data (> {
                  len(data)} years)")
        else:
            print(f"{percentile} Percentile Inflow: Return Period = {
                  return_period:.2f} years")
else:
    print(f"No fitted parameters found for {
          best_fit_name_chi}. Cannot estimate return periods.")

# Visualization of each distribution and test
# Plot histogram and PDFs
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=50, kde=False, stat='density',
             label='Data', color='skyblue')
x = np.linspace(data.min(), data.max(), 1000)
for name, dist in distributions.items():
    params = fitted_params.get(name)
    if params:
        if name == 'Gamma':
            continue
        y = dist.pdf(x, *params)
        plt.plot(x, y, label=name)
plt.plot(x, stats.gamma.pdf(x, *fitted_params.get('Gamma')), label='Gamma')

plt.legend()
plt.title('Histogram and Fitted PDFs')
plt.xlabel('Inflow (m³/s)')
plt.ylabel('Density')
plt.show()

# Plot Empirical CDF and Fitted CDFs
plt.figure(figsize=(10, 6))
# Empirical CDF
sorted_data = np.sort(data)
ecdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
plt.step(sorted_data, ecdf, where='post', label='Empirical CDF')

for name, dist in distributions.items():
    params = fitted_params.get(name)
    if params:
        if name == 'Gamma':
            continue
        cdf_fitted = dist.cdf(sorted_data, *params)
        plt.plot(sorted_data, cdf_fitted, label=f'{name} CDF')
plt.plot(sorted_data, stats.gamma.cdf(
    sorted_data, *fitted_params.get('Gamma')), label='Gamma CDF')

plt.legend()
plt.title('Empirical CDF and Fitted CDFs')
plt.xlabel('Inflow (m³/s)')
plt.ylabel('Cumulative Probability')
plt.show()
