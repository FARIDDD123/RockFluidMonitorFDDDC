import os
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
from pathlib import Path

def setup_directories():
    """ایجاد پوشه‌های مورد نیاز"""
    Path("datasets/processed").mkdir(exist_ok=True)
    Path("datasets/outliers").mkdir(exist_ok=True)
    Path("datasets/stats").mkdir(exist_ok=True)

def detect_outliers(df, method='iqr', threshold=3):
    """
    شناسایی outlierها با روش‌های مختلف
    پارامترها:
        method: 'iqr' یا 'zscore'
        threshold: حد آستانه برای outlierها
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'iqr':
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[numeric_cols] < (Q1 - threshold*IQR)) | 
                    (df[numeric_cols] > (Q3 + threshold*IQR))).any(axis=1)
    
    elif method == 'zscore':
        z_scores = stats.zscore(df[numeric_cols], nan_policy='omit')
        outliers = (np.abs(z_scores) > threshold).any(axis=1)
    
    return outliers

def standardize_data(df, method='standard'):
    """
    استانداردسازی داده‌ها
    پارامترها:
        method: 'standard' (mean-std) یا 'minmax'
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'standard':
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        stats = {
            'method': 'standard',
            'mean': df[numeric_cols].mean().to_dict(),
            'std': df[numeric_cols].std().to_dict()
        }
    elif method == 'minmax':
        min_vals = df[numeric_cols].min()
        max_vals = df[numeric_cols].max()
        df[numeric_cols] = (df[numeric_cols] - min_vals) / (max_vals - min_vals)
        stats = {
            'method': 'minmax',
            'min': min_vals.to_dict(),
            'max': max_vals.to_dict()
        }
    
    return df, stats

def process_partition(df, outlier_method='iqr', standardization_method='standard'):
    """پردازش هر پارتیشن داده"""
    # 1. حذف مقادیر گم‌شده
    df = df.dropna()
    
    # 2. شناسایی outlierها
    outliers = detect_outliers(df, method=outlier_method)
    
    # 3. استانداردسازی داده‌ها
    df_clean, scaling_stats = standardize_data(df[~outliers], method=standardization_method)
    
    return {
        'clean': df_clean,
        'outliers': df[outliers],
        'stats': scaling_stats
    }

def main():
    setup_directories()
    
    with Client(n_workers=4, memory_limit='4GB') as client:
        # خواندن داده‌ها
        ddf = dd.read_parquet("datasets/*.parquet", chunksize=100000)
        
        # پردازش موازی
        results = ddf.map_partitions(
            process_partition,
            outlier_method='zscore',  # یا 'iqr'
            standardization_method='standard'  # یا 'minmax'
        ).compute()
        
        # تجمیع نتایج
        df_clean = dd.concat([r['clean'] for r in results])
        df_outliers = dd.concat([r['outliers'] for r in results])
        
        # ذخیره‌سازی
        df_clean.to_parquet("data/processed/", partition_on=['WELL_ID'])
        df_outliers.to_parquet("data/outliers/")
        
        # ذخیره آماره‌های استانداردسازی
        with open("data/stats/scaling_stats.json", "w") as f:
            json.dump(results[0]['stats'], f)  # آماره‌های اولین پارتیشن به عنوان نمونه

if __name__ == '__main__':
    main()
