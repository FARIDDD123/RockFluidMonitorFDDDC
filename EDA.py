import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
from ydata_profiling import ProfileReport
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
pd.set_option('display.max_columns', 50)

def comprehensive_eda(file_path, output_dir="eda_results"):
    """تحلیل اکتشافی جامع برای داده‌های حفاری"""
    # 1. تنظیمات اولیه
    from pathlib import Path
    output_dir = "output"
    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/plots").mkdir(exist_ok=True)
    
    # 2. بارگذاری داده‌ها
    print("🔍 در حال بارگذاری داده‌ها...")
    df = pd.read_parquet(file_path)
    print(f"✅ داده با {len(df)} رکورد و {len(df.columns)} ویژگی بارگذاری شد.")
    
    # 3. تحلیل ساختار داده‌ها
    print("\n📊 تحلیل ساختار داده‌ها:")
    print("🔹 اطلاعات کلی:")
    print(df.info())
    
    # 4. تحلیل آماری
    stats_report = df.describe(include='all').T
    stats_report['missing_%'] = (df.isnull().sum() / len(df)) * 100
    stats_report.to_csv(f"{output_dir}/statistical_report.csv")
    
    # 5. تحلیل مقادیر گمشده
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Analysis')
    plt.savefig(f"{output_dir}/plots/missing_values.png")
    plt.close()
    
    # 6. تحلیل توزیع ویژگی‌های عددی
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        try:
            plt.figure(figsize=(12, 6))
            sns.histplot(df[col], kde=True, bins=50)
            plt.title(f'Distribution of {col}')
            plt.savefig(f"{output_dir}/plots/dist_{col}.png")
            plt.close()
        except:
            continue
    
    # 7. شناسایی outlierها
    def detect_outliers(df, columns, method='iqr'):
        outliers_report = {}
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR))]
            elif method == 'zscore':
                z_scores = stats.zscore(df[col].dropna())
                outliers = df[(np.abs(z_scores) > 3)]
            
            outliers_report[col] = {
                'count': len(outliers),
                'percentage': (len(outliers)/len(df))*100,
                'min_outlier': outliers[col].min() if len(outliers) > 0 else None,
                'max_outlier': outliers[col].max() if len(outliers) > 0 else None
            }
            
            # رسم boxplot
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=df[col])
            plt.title(f'Boxplot for {col}')
            plt.savefig(f"{output_dir}/plots/boxplot_{col}.png")
            plt.close()
        
        return pd.DataFrame(outliers_report).T
    
    outliers_iqr = detect_outliers(df, numeric_cols, 'iqr')
    outliers_zscore = detect_outliers(df, numeric_cols, 'zscore')
    outliers_iqr.to_csv(f"{output_dir}/outliers_iqr_report.csv")
    outliers_zscore.to_csv(f"{output_dir}/outliers_zscore_report.csv")
    
    # 8. تحلیل همبستگی
    plt.figure(figsize=(20, 15))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.savefig(f"{output_dir}/plots/correlation_matrix.png")
    plt.close()
    
    # 9. تحلیل ویژگی‌های دسته‌ای
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.savefig(f"{output_dir}/plots/cat_dist_{col}.png")
        plt.close()
        
        # تحلیل رابطه با متغیرهای هدف
        if 'Formation_Damage_Index' in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=col, y='Formation_Damage_Index', data=df)
            plt.title(f'Formation Damage Index by {col}')
            plt.savefig(f"{output_dir}/plots/target_{col}.png")
            plt.close()
    
    # 10. تحلیل هدف‌محور
    target_vars = ['Emulsion_Risk', 'Fluid_Loss_Risk', 'Formation_Damage_Index']
    for target in target_vars:
        if target in df.columns:
            # تحلیل رابطه با ویژگی‌های عددی
            top_corr = corr[target].sort_values(ascending=False)
            top_corr.to_csv(f"{output_dir}/top_correlations_{target}.csv")
            
            # تحلیل توزیع
            plt.figure(figsize=(12, 6))
            sns.histplot(df[target], kde=True, bins=50)
            plt.title(f'Distribution of {target}')
            plt.savefig(f"{output_dir}/plots/target_dist_{target}.png")
            plt.close()
    
    # 11. تحلیل زمانی (در صورت وجود ستون زمانی)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        plt.figure(figsize=(15, 8))
        df.groupby('day')['Formation_Damage_Index'].mean().plot()
        plt.title('Daily Average Formation Damage Index')
        plt.savefig(f"{output_dir}/plots/time_analysis.png")
        plt.close()
    
    # 12. تولید گزارش HTML
    print("\n📝 در حال تولید گزارش جامع...")
    profile = ProfileReport(df, title="Drilling Data EDA Report", explorative=True)
    profile.to_file(f"{output_dir}/drilling_eda_report.html")
    
    # 13. خلاصه‌یافته‌ها و توصیه‌ها
    recommendations = generate_recommendations(df, output_dir)
    
    print(f"\n🎉 تحلیل اکتشافی با موفقیت انجام شد! نتایج در پوشه '{output_dir}' ذخیره شد.")
    return df.head(), recommendations

def generate_recommendations(df, output_dir):
    """تولید توصیه‌های مبتنی بر تحلیل داده‌ها"""
    recs = []
    
    # تحلیل مقادیر گمشده
    missing_cols = df.isnull().sum()[df.isnull().sum() > 0].index.tolist()
    if missing_cols:
        recs.append("🚨 ستون‌های دارای مقادیر گمشده: " + ", ".join(missing_cols))
        recs.append("✅ پیشنهاد: پر کردن مقادیر گمشده با میانه (برای داده‌های عددی) یا مد (برای داده‌های دسته‌ای)")
    
    # تحلیل outlierها
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].skew() > 3:
            recs.append(f"⚠️ ستون {col} دارای توزیع بسیار اریب است (Skewness: {df[col].skew():.2f})")
            recs.append(f"✅ پیشنهاد: اعتباریابی مجدد داده‌ها یا استفاده از تبدیل‌های لگاریتمی")
    
    # تحلیل همبستگی
    corr = df[numeric_cols].corr().abs()
    high_corr = [(col1, col2, corr.loc[col1, col2]) 
                for col1 in corr.columns for col2 in corr.columns 
                if col1 < col2 and corr.loc[col1, col2] > 0.8]
    if high_corr:
        recs.append("🔗 ویژگی‌های با همبستگی بالا (>0.8):")
        for pair in high_corr:
            recs.append(f"   - {pair[0]} و {pair[1]}: {pair[2]:.2f}")
        recs.append("✅ پیشنهاد: بررسی multicollinearity و حذف یکی از ویژگی‌های همبسته")
    
    # تحلیل ویژگی‌های هدف
    if 'Formation_Damage_Index' in df.columns:
        target_corr = corr['Formation_Damage_Index'].sort_values(ascending=False)
        top_features = target_corr.index[1:6].tolist()
        recs.append(f"🎯 مهم‌ترین ویژگی‌های تاثیرگذار بر Formation_Damage_Index: {', '.join(top_features)}")
    
    # ذخیره توصیه‌ها
    with open(f"{output_dir}/recommendations.txt", "w") as f:
        f.write("\n".join(recs))
    
    return recs

# اجرای تحلیل
file_path = "synthetic_fdms_chunks/FDMS_well_WELL_1.parquet"
sample_data, recommendations = comprehensive_eda(file_path)

# نمایش نمونه‌ای از داده‌ها و توصیه‌ها
print("\nنمونه‌ای از داده‌ها:")
display(sample_data)

print("\nتوصیه‌های کلیدی:")
for rec in recommendations[:10]:  # نمایش 10 توصیه اول
    print(f"- {rec}")
