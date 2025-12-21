import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, 'reports')
processed_path = os.path.join(ROOT, 'data', 'processed', 'fraud_data_processed.csv')

print('Checking processed path:', processed_path)
df = pd.read_csv(processed_path)
print('Columns:', df.columns.tolist())

# class plot
class_plot_path = os.path.join(REPORTS, 'class_distribution_test.png')
plt.figure(figsize=(6,4))
sns.countplot(x='class', data=df)
plt.title('Fraud (1) vs Legit (0)')
plt.tight_layout()
plt.savefig(class_plot_path, dpi=150)
plt.close()
print('Saved class plot to', class_plot_path)

# purchase value hist
if 'purchase_value' in df.columns:
    pv_plot_path = os.path.join(REPORTS, 'purchase_value_hist_test.png')
    plt.figure(figsize=(8,4))
    sns.histplot(df['purchase_value'].dropna(), bins=50, kde=False)
    plt.title('Purchase Value Distribution')
    plt.tight_layout()
    plt.savefig(pv_plot_path, dpi=150)
    plt.close()
    print('Saved purchase value plot to', pv_plot_path)
else:
    print('purchase_value column not found')
