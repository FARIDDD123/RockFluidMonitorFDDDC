import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

input_dir = "fdms_well_datasets"

def generate_dates(num_rows):
    base_date = datetime(2020, 1, 1)
    spud_dates = [base_date + timedelta(days=int(np.random.uniform(0, 365*3))) for _ in range(num_rows)]
    completion_dates = [spud + timedelta(days=int(np.random.uniform(5, 90))) for spud in spud_dates]
    return spud_dates, completion_dates

for filename in os.listdir(input_dir):
    if filename.endswith(".parquet"):
        filepath = os.path.join(input_dir, filename)
        print(f"📂 در حال پردازش فایل: {filename}")

        df = pd.read_parquet(filepath)

        spud_dates, completion_dates = generate_dates(len(df))
        df["spud_date"] = spud_dates
        df["completion_date"] = completion_dates

        # مرتب‌سازی ستون‌ها: WELL_ID → spud_date → completion_date → باقی ستون‌ها
        first_cols = ["WELL_ID", "spud_date", "completion_date"]
        remaining_cols = [col for col in df.columns if col not in first_cols]
        df = df[first_cols + remaining_cols]

        df.to_parquet(filepath, index=False)
        print(f"✅ ستون‌ها افزوده و فایل ذخیره شد: {filename}")
