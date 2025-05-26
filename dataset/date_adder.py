import pandas as pd
import os

input_dir = "fdms_well_datasets"

for filename in os.listdir(input_dir):
    if filename.endswith(".parquet"):
        filepath = os.path.join(input_dir, filename)
        print(f"⏳ در حال بارگذاری و افزودن timestamp به: {filename}")

        df = pd.read_parquet(filepath)

        # ایجاد ستون timestamp با فاصله 1 ثانیه بین هر ردیف
        df = df.reset_index(drop=True)
        df['timestamp'] = pd.to_datetime('2023-01-01 00:00:00') + pd.to_timedelta(df.index, unit='s')

        # جابجایی ستون timestamp به جایگاه دوم (بعد از WELL_ID)
        cols = df.columns.tolist()
        cols.remove('timestamp')
        well_id_index = cols.index('WELL_ID')
        # ستون timestamp را بعد از WELL_ID اضافه می‌کنیم
        new_cols = cols[:well_id_index + 1] + ['timestamp'] + cols[well_id_index + 1:]
        df = df[new_cols]

        # ذخیره مجدد با ستون جدید timestamp
        df.to_parquet(filepath, index=False)

        print(f"✅ ستون timestamp اضافه و فایل ذخیره شد: {filename}")
