import os
import dask.dataframe as dd
from dask.distributed import Client

def main():
    # کد اصلی پردازش داده‌های FDMS
    client = Client()
    # ... بقیه کدها

if __name__ == '__main__':
    # این بخش فقط در اجرای مستقیم فایل فعال می‌شود
    main()
import json
from pathlib import Path

# 1. تنظیم مسیرها
DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTLIER_DIR = "data/outliers"
STATS_DIR = "data/stats"

# ایجاد پوشه‌ها
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTLIER_DIR).mkdir(parents=True, exist_ok=True)
Path(STATS_DIR).mkdir(parents=True, exist_ok=True)

# 2. راه‌اندازی پردازش موازی
client = Client(n_workers=4, memory_limit='4GB')  # تنظیم براساس RAM سیستم شما

# 3. خواندن داده‌ها با Dask
print("در حال خواندن داده‌ها...")
ddf = dd.read_parquet(os.path.join(DATA_DIR, "*.parquet"))

# 4. پردازش اولیه
def preprocess(df):
    # حذف داده‌های گم‌شده
    df = df.dropna()
    
    # محاسبه outlierها
    numeric_cols = df.select_dtypes(include=['number']).columns
    q1 = df[numeric_cols].quantile(0.25)
    q3 = df[numeric_cols].quantile(0.75)
    iqr = q3 - q1
    
    outliers = ((df[numeric_cols] < (q1 - 1.5*iqr)) | (df[numeric_cols] > (q3 + 1.5*iqr))).any(axis=1)
    
    return {
        'clean': df[~outliers],
        'outliers': df[outliers],
        'stats': {
            'min': df[numeric_cols].min(),
            'max': df[numeric_cols].max()
        }
    }

# 5. اجرای پردازش
print("در حال پردازش داده‌ها...")
result = ddf.map_partitions(preprocess).compute()

# 6. ذخیره‌سازی نتایج
print("ذخیره نتایج...")
result['clean'].to_parquet(PROCESSED_DIR, engine='pyarrow')
result['outliers'].to_parquet(OUTLIER_DIR, engine='pyarrow')

with open(os.path.join(STATS_DIR, "scaling_stats.json"), 'w') as f:
    json.dump(result['stats'], f)

print("✅ پردازش با موفقیت انجام شد!")
client.close()

ddf = dd.read_parquet(..., chunksize=100_000)

client = Client(n_workers=2, memory_limit='2GB')

ddf.to_parquet(..., partition_on=['WELL_ID'])

from dask.diagnostics import ProgressBar
with ProgressBar():
    result = ddf.map_partitions(preprocess).compute()

def process_fdms_data():
    """تابع اصلی پردازش داده‌های FDMS"""
    client = Client()
    try:
        # کد پردازش داده‌های چاه
        df = dd.read_parquet("data/raw/*.parquet")
        # ... عملیات پردازش
    finally:
        client.close()

if __name__ == '__main__':
    process_fdms_data()  # حالا دیگر خطا نمی‌دهد
if __name__ == '__main__':
    client = Client()
    try:
        process_fdms_data()  # تابع اصلی شما
    finally:
        client.close()

#outliers
#IQR & z-score
#standardization
#mean std z-score

#5
#5
#1
#1
