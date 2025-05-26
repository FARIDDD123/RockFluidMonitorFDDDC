import pandas as pd
import numpy as np
import os

np.random.seed(42)

num_rows_per_well = 15_552_000

well_info = [
    {"WELL_ID": 50, "LAT": 32.26, "LONG": -94.86},
    {"WELL_ID": 31881, "LAT": 32.26, "LONG": -94.82},
    {"WELL_ID": 34068, "LAT": 32.25, "LONG": -94.78},
    {"WELL_ID": 81715, "LAT": 32.17, "LONG": -94.95},
    {"WELL_ID": 35068, "LAT": 32.32, "LONG": -94.18},
    {"WELL_ID": 362, "LAT": 32.32, "LONG": -94.13},
    {"WELL_ID": 30944, "LAT": 32.12, "LONG": -94.15},
    {"WELL_ID": 32094, "LAT": 32.37, "LONG": -94.62},
    {"WELL_ID": 31921, "LAT": 32.37, "LONG": -94.6},
    {"WELL_ID": 87931, "LAT": 32.45, "LONG": -94.86},
]

formation_types = ['Shale', 'Sandstone', 'Limestone', 'Dolomite', 'Chalk']
bit_types = ['PDC', 'Roller Cone', 'Hybrid']
shale_reactivity = ['Low', 'Medium', 'High']

numeric_cols = [
    "Depth_m", "ROP_mph", "WOB_kgf", "Torque_Nm", "Pump_Pressure_psi",
    "Mud_FlowRate_LPM", "MWD_Vibration_g", "Mud_Weight_ppg", "Viscosity_cP",
    "Plastic_Viscosity", "Yield_Point", "pH_Level", "Solid_Content_%",
    "Chloride_Concentration_mgL", "Oil_Water_Ratio", "Emulsion_Stability",
    "Pore_Pressure_psi", "Fracture_Gradient_ppg", "Stress_Tensor_MPa",
    "Young_Modulus_GPa", "Poisson_Ratio", "Brittleness_Index"
]

def fluid_loss_risk(row):
    if row['Mud_Weight_ppg'] > row['Fracture_Gradient_ppg']:
        return 'High'
    elif row['Mud_Weight_ppg'] > row['Fracture_Gradient_ppg'] - 0.5:
        return 'Medium'
    else:
        return 'Low'

def emulsion_risk(row):
    return np.clip((row['Oil_Water_Ratio'] / 100 + row['Solid_Content_%'] / 30), 0, 1)

def reactivity_score(row):
    if row['Shale_Reactiveness'] == 'High':
        return np.random.uniform(0.6, 1.0)
    elif row['Shale_Reactiveness'] == 'Medium':
        return np.random.uniform(0.3, 0.6)
    else:
        return np.random.uniform(0.0, 0.3)

output_dir = "fdms_well_datasets"
os.makedirs(output_dir, exist_ok=True)

for i, info in enumerate(well_info):
    print(f"⏳ در حال تولید داده برای چاه {info['WELL_ID']} ...")

    shift = i * 0.1
    scale = 1 + (i % 5) * 0.05

    df = pd.DataFrame({
        "Depth_m": np.random.normal(3000 + shift*500, 800 * scale, num_rows_per_well).clip(1000, 6000),
        "ROP_mph": np.random.normal(20 + shift*2, 8 * scale, num_rows_per_well).clip(5, 50),
        "WOB_kgf": np.random.normal(15000 + shift*1000, 5000 * scale, num_rows_per_well).clip(5000, 30000),
        "Torque_Nm": np.random.normal(1000 + shift*50, 400 * scale, num_rows_per_well).clip(200, 2000),
        "Pump_Pressure_psi": np.random.normal(3000 + shift*500, 1000 * scale, num_rows_per_well).clip(1000, 6000),
        "Mud_FlowRate_LPM": np.random.uniform(100 + shift*20, 800 + shift*30, num_rows_per_well),
        "MWD_Vibration_g": np.random.uniform(0.1, 3.0 + shift, num_rows_per_well),
        "Bit_Type": np.random.choice(bit_types, num_rows_per_well),
        "Mud_Weight_ppg": np.random.normal(11 + shift, 1.5 * scale, num_rows_per_well).clip(8.5, 15),
        "Viscosity_cP": np.random.normal(70 + shift*5, 20 * scale, num_rows_per_well).clip(30, 120),
        "Plastic_Viscosity": np.random.normal(30 + shift*2, 8 * scale, num_rows_per_well).clip(10, 50),
        "Yield_Point": np.random.normal(20 + shift*2, 6 * scale, num_rows_per_well).clip(5, 40),
        "pH_Level": np.random.normal(8.5, 1.2 * scale, num_rows_per_well).clip(6.5, 11),
        "Solid_Content_%": np.random.uniform(1, 20, num_rows_per_well),
        "Chloride_Concentration_mgL": np.random.normal(50000 + shift*5000, 20000 * scale, num_rows_per_well).clip(100, 150000),
        "Oil_Water_Ratio": np.random.uniform(10, 90, num_rows_per_well),
        "Emulsion_Stability": np.random.uniform(30, 100, num_rows_per_well),
        "Formation_Type": np.random.choice(formation_types, num_rows_per_well),
        "Pore_Pressure_psi": np.random.normal(8000 + shift*500, 2000 * scale, num_rows_per_well).clip(3000, 15000),
        "Fracture_Gradient_ppg": np.random.normal(15 + shift*0.2, 1.5 * scale, num_rows_per_well).clip(13, 18),
        "Stress_Tensor_MPa": np.random.normal(40 + shift*2, 15 * scale, num_rows_per_well).clip(10, 80),
        "Young_Modulus_GPa": np.random.normal(30 + shift*3, 10 * scale, num_rows_per_well).clip(5, 70),
        "Poisson_Ratio": np.random.uniform(0.2, 0.35, num_rows_per_well),
        "Brittleness_Index": np.random.uniform(0, 1, num_rows_per_well),
        "Shale_Reactiveness": np.random.choice(shale_reactivity, num_rows_per_well),
    })

    df["Fluid_Loss_Risk"] = df.apply(fluid_loss_risk, axis=1)
    df["Emulsion_Risk"] = df.apply(emulsion_risk, axis=1)
    df["Rock_Fluid_Reactivity"] = df.apply(reactivity_score, axis=1)
    df["Formation_Damage_Index"] = (
        df["Emulsion_Risk"] * 0.4 +
        df["Rock_Fluid_Reactivity"] * 0.3 +
        df["Brittleness_Index"] * 0.2 +
        np.random.normal(0, 0.05, num_rows_per_well)
    )

    outlier_fraction = 0.05
    num_outliers = int(outlier_fraction * num_rows_per_well)
    outlier_indices = np.random.choice(df.index, num_outliers, replace=False)

    for col in numeric_cols:
        df.loc[outlier_indices[:num_outliers//2], col] *= np.random.uniform(3, 5)
        df.loc[outlier_indices[num_outliers//2:], col] *= np.random.uniform(0.1, 0.5)

    missing_fraction = 0.03
    num_missing = int(missing_fraction * num_rows_per_well)
    missing_indices = np.random.choice(df.index, num_missing, replace=False)
    missing_cols = np.random.choice(numeric_cols, size=5, replace=False)

    for col in missing_cols:
        df.loc[missing_indices, col] = np.nan

    df["WELL_ID"] = info["WELL_ID"]
    df["LAT"] = info["LAT"]
    df["LONG"] = info["LONG"]

    # اضافه کردن ستون timestamp با فواصل 1 ثانیه از تاریخ اولیه
    start_time = pd.to_datetime('2023-01-01 00:00:00')
    df = df.reset_index(drop=True)
    df['timestamp'] = start_time + pd.to_timedelta(df.index, unit='s')

    # مرتب سازی ستون‌ها به نحوی که timestamp بعد از WELL_ID بیاید و LAT و LONG هم بمانند
    cols = df.columns.tolist()
    # حذف ستون timestamp از لیست فعلی
    cols.remove('timestamp')
    # قرار دادن timestamp بعد از WELL_ID
    well_id_index = cols.index('WELL_ID')
    new_cols = cols[:well_id_index + 1] + ['timestamp'] + cols[well_id_index + 1:]
    df = df[new_cols]

    # ذخیره‌ی دیتاست برای هر چاه
    filename = f"FDMS_well_{info['WELL_ID']}.parquet"
    filepath = os.path.join(output_dir, filename)
    df.to_parquet(filepath, index=False)

    print(f"✅ دیتاست ذخیره شد: {filepath}")
