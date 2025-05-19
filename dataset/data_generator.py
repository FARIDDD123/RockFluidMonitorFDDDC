import pandas as pd
import numpy as np

np.random.seed(42)

num_rows_per_well = 1_000_000

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

dfs = []

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

for info in well_info:
    df = pd.DataFrame({
        "Depth_m": np.random.uniform(1000, 6000, num_rows_per_well),
        "ROP_mph": np.random.uniform(5, 50, num_rows_per_well),
        "WOB_kgf": np.random.uniform(5000, 30000, num_rows_per_well),
        "Torque_Nm": np.random.uniform(200, 2000, num_rows_per_well),
        "Pump_Pressure_psi": np.random.uniform(1000, 6000, num_rows_per_well),
        "Mud_FlowRate_LPM": np.random.uniform(100, 800, num_rows_per_well),
        "MWD_Vibration_g": np.random.uniform(0.1, 3.0, num_rows_per_well),
        "Bit_Type": np.random.choice(bit_types, num_rows_per_well),
        "Mud_Weight_ppg": np.random.uniform(8.5, 15.0, num_rows_per_well),
        "Viscosity_cP": np.random.uniform(30, 120, num_rows_per_well),
        "Plastic_Viscosity": np.random.uniform(10, 50, num_rows_per_well),
        "Yield_Point": np.random.uniform(5, 40, num_rows_per_well),
        "pH_Level": np.random.uniform(6.5, 11.0, num_rows_per_well),
        "Solid_Content_%": np.random.uniform(1, 20, num_rows_per_well),
        "Chloride_Concentration_mgL": np.random.uniform(100, 150000, num_rows_per_well),
        "Oil_Water_Ratio": np.random.uniform(10, 90, num_rows_per_well),
        "Emulsion_Stability": np.random.uniform(30, 100, num_rows_per_well),
        "Formation_Type": np.random.choice(formation_types, num_rows_per_well),
        "Pore_Pressure_psi": np.random.uniform(3000, 15000, num_rows_per_well),
        "Fracture_Gradient_ppg": np.random.uniform(13, 18, num_rows_per_well),
        "Stress_Tensor_MPa": np.random.uniform(10, 80, num_rows_per_well),
        "Young_Modulus_GPa": np.random.uniform(5, 70, num_rows_per_well),
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

    # جابجایی ستون‌ها: WELL_ID, LAT, LONG ابتدا
    cols = ['WELL_ID', 'LAT', 'LONG'] + [col for col in df.columns if col not in ['WELL_ID', 'LAT', 'LONG']]
    df = df[cols]

    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

# ذخیره
output_path = "FDMS_synthetic_dataset_10M.parquet"
final_df.to_parquet(output_path, index=False)

print(f"✅ فایل نهایی با ستون‌های موقعیت در ابتدا ذخیره شد: {output_path}")
