import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["SD", "MRMR","GA","Weight Matrix"]
features = [
    ['normal_max_F7', 'normal_rms_F7', 'normal_dasdv_F7', 'normal_krt_F7', 'normal_mean_F7', 'normal_mav_F7', 'normal_mean_second_diff_F7', 'normal_hurst_F7', 'normal_shannon_F7', 'normal_aac_F7', 'normal_log_F7', 'normal_std_F7'],
    ['normal_fuzzy_F7', 'normal_mean_first_diff_F7', 'normal_skw_F7', 'normal_hurst_F7', 'normal_min_F7', 'normal_iqr_F7', 'normal_dfa_F7', 'normal_pe_F7', 'normal_log_F7', 'normal_hjorth_F7', 'normal_shannon_F7', 'normal_abssum_F7'],
    ['normal_mean_F7', 'normal_max_F7', 'normal_min_F7', 'normal_iqr_F7', 'normal_var_F7', 'normal_skw_F7', 'normal_mean_first_diff_F7', 'normal_mean_second_diff_F7', 'normal_rms_F7', 'normal_krt_F7', 'normal_std_F7', 'normal_sum_F7'],
    ['normal_shannon_F7', 'normal_krt_F7', 'normal_pe_F7', 'normal_log_F7', 'normal_mean_F7', 'normal_mav_F7', 'normal_dasdv_F7', 'normal_fuzzy_F7', 'normal_min_F7', 'normal_skw_F7', 'normal_dfa_F7', 'normal_hurst_F7']
]

metrics = {
    "Accuracy": [0.40, 0.39, 0.44,0.40],
    "Precision": [0.41, 0.38, 0.41,0.38],
    "Recall": [0.40, 0.39, 0.43, 0.40],
    "F1 Score": [0.38, 0.36, 0.40, 0.36],
    "Specificity": [0.70, 0.70, 0.72,0.70],
    "Geometric Mean": [0.53, 0.52, 0.55, 0.53]
}

# Plotting the first figure: Top 12 features for each method
fig1, ax1 = plt.subplots(figsize=(12, 6))
for i, method in enumerate(methods):
    ax1.scatter([method] * 12, features[i], marker='o')

ax1.set_title("Top 12 features selected by each method")
ax1.set_ylabel("Feature")
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.set_axisbelow(True)
plt.tight_layout()
plt.show()

# Plotting the second figure: Performance Metrics Comparison
x = np.arange(len(methods))
bar_width = 0.15
fig2, ax2 = plt.subplots(figsize=(12, 6))
for idx, (metric_name, values) in enumerate(metrics.items()):
    ax2.bar(x + idx * bar_width, values, bar_width, label=metric_name)

ax2.set_title("Performance Metrics Comparison")
ax2.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
ax2.set_xticklabels(methods)
ax2.set_ylabel("Score")
ax2.set_ylim([0, 1])
ax2.legend()
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.set_axisbelow(True)
plt.tight_layout()
plt.show()
