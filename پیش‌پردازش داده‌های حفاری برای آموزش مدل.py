import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from pathlib import Path

# مسیر پوشه‌ها
DATA_DIR = "dataset/fdms_well_datasets"
PROCESSED_DIR = "dataset/processed"
OUTLIER_DIR = "dataset/outliers"
MEAN_DIR = "dataset/means"

# ساخت مسیرها
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTLIER_DIR).mkdir(parents=True, exist_ok=True)
Path(MEAN_DIR).mkdir(parents=True, exist_ok=True)

# پردازش هر فایل
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".parquet"):
        well_id = filename.split("_")[-1].replace(".parquet", "")
        raw_path = os.path.join(DATA_DIR, filename)
        processed_path = os.path.join(PROCESSED_DIR, f"cleaned_{well_id}.parquet")
        outliers_path = os.path.join(OUTLIER_DIR, f"outliers_{well_id}.parquet")
        mean_json_path = os.path.join(MEAN_DIR, f"mean_{well_id}.json")

        print(f"\n📥 Loading data for well {well_id} from {raw_path}")
        df = pd.read_parquet(raw_path)

        print(f"📊 Initial shape: {df.shape}")
        print("🔍 Missing values:\n", df.isna().sum())

        # حذف NaN
        df = df.dropna()

        # انتخاب ویژگی‌های عددی
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['WELL_ID', 'LAT', 'LONG']]

        # استخراج min و max
        original_min = df[numeric_cols].min().to_dict()
        original_max = df[numeric_cols].max().to_dict()

        # نرمال‌سازی
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # دسته‌بندی ویژگی‌های متنی
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        df = pd.get_dummies(df, columns=categorical_cols)

        # شناسایی داده‌های پرت
        z_scores = np.abs(stats.zscore(df[numeric_cols]))
        outliers_z = (z_scores > 3).any(axis=1)

        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

        outliers = outliers_z | outliers_iqr
        print(f"🚨 Outliers detected: {outliers.sum()} rows")

        # ذخیره داده‌های پرت
        df_outliers = df[outliers]
        df_outliers.to_parquet(outliers_path, index=False)

        # ذخیره داده تمیز
        df_clean = df[~outliers]
        df_clean.to_parquet(processed_path, index=False)

        # ذخیره آماره‌ها برای برگرداندن نرمال‌سازی
        stats_to_save = {
            "WELL_ID": int(well_id),
            "scaling_method": "minmax",
            "min": original_min,
            "max": original_max
        }

        with open(mean_json_path, 'w') as f:
            json.dump(stats_to_save, f, indent=4)

        print(f"✅ Preprocessing complete and stats saved → well {well_id}")
