"""Generate a PDF report for the project.

Usage:
    python scripts/generate_pdf_report.py

Creates a PDF in the `reports/` folder named safely for Windows:
    B8W5 - Improved detection of fraud cases for e-commerce and bank - final Submission - Report.pdf

This script installs `reportlab` locally if not available.
"""
import os
import sys
import subprocess
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.enums import TA_CENTER
except Exception:
    print('reportlab not found, installing...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'reportlab'])
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.enums import TA_CENTER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT, 'reports')
if not os.path.isdir(REPORTS_DIR):
    os.makedirs(REPORTS_DIR, exist_ok=True)

# Windows-safe filename (remove colon and other illegal chars)
base_name = 'B8W5 - Improved detection of fraud cases for e-commerce and bank - final Submission - Report'
safe_name = ''.join(c if c not in '\\/:*?"<>|' else '-' for c in base_name) + '.pdf'
out_path = os.path.join(REPORTS_DIR, safe_name)

# Read contents to include
readme_path = os.path.join(ROOT, 'README.md')
shap_md_path = os.path.join(REPORTS_DIR, 'SHAP_explainability.md')

def read_file(path, max_lines=None):
    if not os.path.exists(path):
        return ''
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if max_lines:
        lines = lines[:max_lines]
    return ''.join(lines)

readme = read_file(readme_path, max_lines=400)
shap = read_file(shap_md_path, max_lines=400)

# Images to attach if present
image_files = []
for img in ['feature_importance_top10.png', 'shap_summary.png']:
    img_path = os.path.join(REPORTS_DIR, img)
    if os.path.exists(img_path):
        image_files.append(img_path)

# Generate simple visuals from processed data if available
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
except Exception:
    print('matplotlib/seaborn/pandas not found, installing...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib', 'seaborn', 'pandas'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

processed_path = os.path.join(ROOT, 'data', 'processed', 'fraud_data_processed.csv')
# Class distribution
class_plot = os.path.join(REPORTS_DIR, 'class_distribution.png')
if os.path.exists(processed_path):
    try:
        df = pd.read_csv(processed_path)
        print('generate_pdf_report: loaded processed data, rows=', len(df), 'cols=', len(df.columns))
        if 'class' in df.columns:
            plt.figure(figsize=(6,4))
            sns.countplot(x='class', data=df)
            plt.title('Fraud (1) vs Legit (0)')
            plt.tight_layout()
            plt.savefig(class_plot, dpi=150)
            plt.close()
            image_files.append(class_plot)
            print('generate_pdf_report: saved', class_plot)
        # Purchase value distribution
        if 'purchase_value' in df.columns:
            pv_plot = os.path.join(REPORTS_DIR, 'purchase_value_hist.png')
            plt.figure(figsize=(8,4))
            sns.histplot(df['purchase_value'].dropna(), bins=50, kde=False)
            plt.title('Purchase Value Distribution')
            plt.tight_layout()
            plt.savefig(pv_plot, dpi=150)
            plt.close()
            image_files.append(pv_plot)
            print('generate_pdf_report: saved', pv_plot)
        # Correlation heatmap for numeric columns (top 10)
        numeric = df.select_dtypes(include='number')
        if numeric.shape[1] >= 2:
            corr = numeric.corr().abs()
            cols = corr.sum().sort_values(ascending=False).index[:10]
            hm_plot = os.path.join(REPORTS_DIR, 'corr_heatmap.png')
            plt.figure(figsize=(8,6))
            sns.heatmap(numeric[cols].corr(), annot=False, cmap='viridis')
            plt.title('Correlation Heatmap (top features)')
            plt.tight_layout()
            plt.savefig(hm_plot, dpi=150)
            plt.close()
            image_files.append(hm_plot)
            print('generate_pdf_report: saved', hm_plot)
    except Exception as e:
        print('Could not generate visuals from processed data:', e)
else:
    print('Processed data not found at', processed_path, '— skipping auto visuals')
# Git info
git_commit = ''
try:
    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT).decode().strip()
    commit_msg = subprocess.check_output(['git', 'log', '-1', '--pretty=format:%s'], cwd=ROOT).decode().strip()
    git_commit = f'{commit_hash} - {commit_msg}'
except Exception:
    git_commit = 'n/a'

# Build PDF
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='CenterTitle', alignment=TA_CENTER, fontSize=18, leading=22))
styles.add(ParagraphStyle(name='SectionHeading', fontSize=14, leading=18))
styles.add(ParagraphStyle(name='Small', fontSize=9, leading=11))

doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
story = []

# Title
story.append(Spacer(1, 20))
story.append(Paragraph('B8W5: Improved detection of fraud cases for e-commerce and bank', styles['CenterTitle']))
story.append(Spacer(1, 6))
story.append(Paragraph('Final Submission - Report', styles['CenterTitle']))
story.append(Spacer(1, 12))
story.append(Paragraph(f'Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}', styles['Small']))
story.append(Paragraph(f'Git commit: {git_commit}', styles['Small']))
story.append(PageBreak())

# README summary
story.append(Paragraph('Project Summary', styles['SectionHeading']))
if readme:
    # Simplified: take first 6 paragraphs
    paragraphs = [p.strip() for p in readme.split('\n\n') if p.strip()]
    for p in paragraphs[:6]:
        story.append(Paragraph(p.replace('\n', ' '), styles['Normal']))
        story.append(Spacer(1, 6))
else:
    story.append(Paragraph('README.md not found in repository root.', styles['Normal']))
story.append(PageBreak())
# Final Project Report (user-provided)
story.append(Paragraph('Final Project Report', styles['SectionHeading']))
final_report_paragraphs = [
    '1. Introduction\nFraudulent financial transactions pose a significant threat to both e-commerce platforms and banking institutions. As digital payments continue to grow, so does the sophistication of fraudulent activities. This project was conducted for Adey Innovations Inc. with the objective of developing accurate, explainable, and business-oriented fraud detection models for e-commerce and bank transactions.',
    '2. Business Problem and Objectives\nThe primary business challenge is balancing fraud prevention with customer experience. False positives can frustrate legitimate users, while false negatives lead to direct financial losses. The objective of this project is to design machine learning models that effectively detect fraud in highly imbalanced datasets while remaining interpretable and suitable for real-time monitoring.',
    '3. Dataset Description\nE-commerce Dataset (Fraud_Data.csv): Contains user, transaction, device, and behavioral features including signup time, purchase time, purchase value, browser, age, and IP address, with a binary fraud label.\nBank Transaction Dataset (creditcard.csv): Contains anonymized PCA-transformed features (V1–V28), transaction amount, and a binary fraud label. Both datasets exhibit severe class imbalance.',
    '4. Data Preprocessing and Feature Engineering\nData preprocessing included handling missing values, correcting data types, removing duplicates, and scaling numerical features. For the e-commerce data, IP addresses were converted to integer format and merged with a country-mapping dataset to enable geolocation analysis.\nKey engineered features included time_since_signup, hour_of_day, day_of_week, and transaction frequency metrics. Categorical variables were encoded, and numerical features were normalized to improve model performance.',
    '5. Handling Class Imbalance\nBoth datasets were highly imbalanced, with fraudulent transactions representing a very small proportion of total observations. To address this, resampling techniques such as SMOTE were applied to the training data only. Evaluation metrics focused on precision, recall, F1-score, and precision-recall AUC rather than accuracy.',
    '6. Model Building and Evaluation\nA Logistic Regression model was used as a baseline due to its interpretability. An ensemble model (such as Random Forest or Gradient Boosting) was then trained to capture complex, non-linear patterns. Stratified train-test splits and cross-validation ensured reliable evaluation.\nThe ensemble model outperformed the baseline by achieving better recall of fraudulent transactions while maintaining reasonable precision, making it more suitable for real-world fraud detection systems.',
    '7. Model Explainability (SHAP Analysis)\nModel explainability was achieved using SHAP (SHapley Additive exPlanations). Global SHAP summary plots revealed the most influential features, while local SHAP force plots explained individual predictions, including true positives, false positives, and false negatives.\nThe top drivers of fraud predictions were time_since_signup, purchase_value, hour_of_day, user behavior patterns, and age. These insights align well with domain knowledge and increase trust in the model’s decisions.',
    '8. Business Insights and Recommendations\nBased on the analysis, transactions occurring shortly after signup should receive additional verification. High-value transactions at unusual hours should be monitored more closely. Explainable fraud detection models should be integrated into operational systems to support compliance and trust.',
    '9. Conclusion\nThis project demonstrates that combining advanced machine learning models with explainable AI techniques can significantly improve fraud detection for both e-commerce and banking transactions. The proposed solution balances predictive performance, interpretability, and business practicality, making it suitable for deployment in real-world financial systems.'
]
for p in final_report_paragraphs:
    story.append(Paragraph(p.replace('\n', ' '), styles['Normal']))
    story.append(Spacer(1, 6))
# Insert dataset visuals (class distribution and purchase value) and feature-engineering visual
class_plot_path = os.path.join(REPORTS_DIR, 'class_distribution.png')
pv_plot_path = os.path.join(REPORTS_DIR, 'purchase_value_hist.png')
hm_plot_path = os.path.join(REPORTS_DIR, 'corr_heatmap.png')
if os.path.exists(class_plot_path):
    story.append(Spacer(1, 6))
    story.append(Paragraph('Dataset Visual: Class Distribution', styles['SectionHeading']))
    try:
        im = Image(class_plot_path)
        max_width = A4[0] - (40*mm)
        if im.drawWidth > max_width:
            ratio = max_width / im.drawWidth
            im.drawWidth = im.drawWidth * ratio
            im.drawHeight = im.drawHeight * ratio
        story.append(im)
    except Exception:
        story.append(Paragraph('Could not include class distribution image.', styles['Small']))
    story.append(Spacer(1, 6))
if os.path.exists(pv_plot_path):
    story.append(Paragraph('Dataset Visual: Purchase Value Distribution', styles['SectionHeading']))
    try:
        im = Image(pv_plot_path)
        max_width = A4[0] - (40*mm)
        if im.drawWidth > max_width:
            ratio = max_width / im.drawWidth
            im.drawWidth = im.drawWidth * ratio
            im.drawHeight = im.drawHeight * ratio
        story.append(im)
    except Exception:
        story.append(Paragraph('Could not include purchase value image.', styles['Small']))
    story.append(Spacer(1, 6))
if os.path.exists(hm_plot_path):
    story.append(Paragraph('Feature Engineering Visual: Correlation Heatmap', styles['SectionHeading']))
    try:
        im = Image(hm_plot_path)
        max_width = A4[0] - (40*mm)
        if im.drawWidth > max_width:
            ratio = max_width / im.drawWidth
            im.drawWidth = im.drawWidth * ratio
            im.drawHeight = im.drawHeight * ratio
        story.append(im)
    except Exception:
        story.append(Paragraph('Could not include correlation heatmap image.', styles['Small']))
    story.append(PageBreak())

# SHAP explainability
story.append(Paragraph('SHAP Explainability', styles['SectionHeading']))
if shap:
    paragraphs = [p.strip() for p in shap.split('\n\n') if p.strip()]
    for p in paragraphs[:10]:
        story.append(Paragraph(p.replace('\n', ' '), styles['Normal']))
        story.append(Spacer(1, 6))
else:
    story.append(Paragraph('SHAP explanation notes not found in reports/SHAP_explainability.md.', styles['Normal']))

# Include SHAP and feature-importance images (avoid duplicating dataset visuals)
shap_images = []
for img in ['shap_summary.png', 'feature_importance_top10.png']:
    img_path = os.path.join(REPORTS_DIR, img)
    if os.path.exists(img_path):
        shap_images.append(img_path)
for img_path in shap_images:
    story.append(Spacer(1, 10))
    story.append(Paragraph(os.path.basename(img_path).replace('_', ' ').replace('.png', ''), styles['SectionHeading']))
    try:
        im = Image(img_path)
        max_width = A4[0] - (40*mm)
        if im.drawWidth > max_width:
            ratio = max_width / im.drawWidth
            im.drawWidth = im.drawWidth * ratio
            im.drawHeight = im.drawHeight * ratio
        story.append(im)
    except Exception as e:
        story.append(Paragraph(f'Could not include image {os.path.basename(img_path)}: {e}', styles['Small']))

# Footer
story.append(PageBreak())
story.append(Paragraph('Appendix: File list', styles['SectionHeading']))
files_list = '\n'.join(sorted(os.listdir(ROOT)))
for line in files_list.split('\n'):
    story.append(Paragraph(line, styles['Normal']))

# Build
try:
    doc.build(story)
    print(f'PDF generated at: {out_path}')
except Exception as e:
    print('Failed to generate PDF:', e)
    sys.exit(1)

# Optionally, print path for convenience
print('Done')
