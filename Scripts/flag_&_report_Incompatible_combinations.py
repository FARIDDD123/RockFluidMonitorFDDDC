import pandas as pd
import numpy as np
import os

# --- تابع‌های شاخص ---
def fluid_loss_risk(row):
    return min(1, (row["Viscosity_cP"] / 120) * (row["Solid_Content_%"] / 20))

def emulsion_risk(row):
    return min(1, (100 - row["Oil_Water_Ratio"]) / 100 + (100 - row["Emulsion_Stability"]) / 100)

def reactivity_score(row):
    if row["Shale_Reactiveness"] == "High":
        return 1
    elif row["Shale_Reactiveness"] == "Medium":
        return 0.5
    return 0

# --- تابع فلگ‌گذاری ترکیب‌های غیرمنطقی ---
def flag_unrealistic_combinations(df):
    df["Fluid_Loss_Risk"] = df.apply(fluid_loss_risk, axis=1)
    df["Emulsion_Risk"] = df.apply(emulsion_risk, axis=1)
    df["Rock_Fluid_Reactivity"] = df.apply(reactivity_score, axis=1)
    df["Formation_Damage_Index"] = (
        df["Emulsion_Risk"] * 0.4 +
        df["Rock_Fluid_Reactivity"] * 0.3 +
        df["Brittleness_Index"] * 0.2 +
        np.random.normal(0, 0.01, len(df))
    )

    # قوانین مهندسی/غیرمنطقی
    df["Flag_Unrealistic_1"] = (
        (df["Viscosity_cP"] > 110) &
        (df["Solid_Content_%"] > 18) &
        (df["pH_Level"] < 7)
    )

    df["Flag_Unrealistic_2"] = (
        (df["Oil_Water_Ratio"] < 20) &
        (df["Emulsion_Stability"] < 40)
    )

    df["Flag_Unrealistic_3"] = (
        (df["Rock_Fluid_Reactivity"] == 1) &
        (df["Chloride_Concentration_mgL"] > 120000)
    )

    df["Flag_Unrealistic_4"] = (
        (df["Brittleness_Index"] > 0.9) &
        (df["Poisson_Ratio"] > 0.33)
    )

    df["Flag_Unrealistic_5"] = (
        (df["Mud_Weight_ppg"] > 14.5) &
        (df["ROP_mph"] > 40)
    )

    df["Flag_Unrealistic_6"] = (
        (df["Stress_Tensor_MPa"] < 15) &
        (df["Young_Modulus_GPa"] > 60)
    )

    # ترکیب کلی
    flag_cols = [col for col in df.columns if col.startswith("Flag_Unrealistic_")]
    df["Flag_Unrealistic_Combo"] = df[flag_cols].any(axis=1)

    return df

# --- تابع گزارش‌گیری آماری ---
def report_unrealistic_summary(df):
    total = len(df)
    flagged = df["Flag_Unrealistic_Combo"].sum()

    summary = {
        "Total_Records": total,
        "Flagged_Records": flagged,
        "Flagged_Percentage": (flagged / total) * 100 if total > 0 else 0
    }

    for col in df.columns:
        if col.startswith("Flag_Unrealistic_") and col != "Flag_Unrealistic_Combo":
            summary[col] = df[col].sum()

    print(f"\n🔍 رکوردهای علامت‌خورده: {flagged:,} از {total:,} ({summary['Flagged_Percentage']:.2f}%)\n")
    for col in summary:
        if col.startswith("Flag_Unrealistic_") and col != "Flag_Unrealistic_Combo":
            print(f"  {col:<25}: {summary[col]:,}")

    return summary

# --- اجرای اصلی روی همه فایل‌ها در پوشه مشخص‌شده و جمع‌آوری گزارش ---
def process_all_files(input_dir, output_dir=None, report_csv_path=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]

    report_rows = []

    for fname in files:
        print(f"\n📂 پردازش فایل: {fname}")
        df = pd.read_parquet(os.path.join(input_dir, fname))

        df = flag_unrealistic_combinations(df)
        summary = report_unrealistic_summary(df)

        # ذخیره خروجی اگر پوشه خروجی داده شده باشه
        if output_dir:
            out_path = os.path.join(output_dir, fname.replace(".parquet", "_flagged.parquet"))
            df.to_parquet(out_path, index=False)
            print(f"✅ فایل ذخیره شد: {out_path}")

        # افزودن ردیف گزارش
        summary["File"] = fname
        report_rows.append(summary)

    # ذخیره گزارش کلی به صورت CSV
    if report_csv_path:
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(report_csv_path, index=False)
        print(f"\n📊 گزارش کلی ذخیره شد در: {report_csv_path}")

# --- استفاده ---
if __name__ == "__main__":
    input_data_path = "synthetic_fdms_chunks"       # ← مسیر فایل‌های ورودی شما
    output_data_path = "flagged_outputs"            # ← مسیر خروجی (اختیاری)
    report_csv_file = "unrealistic_report_summary.csv"  # ← مسیر فایل گزارش CSV
    
    process_all_files(input_data_path, output_data_path, report_csv_file)
