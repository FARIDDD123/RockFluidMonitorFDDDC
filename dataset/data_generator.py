import pandas as pd
import numpy as np
import os

# تابع‌های تولید شاخص‌ها
def fluid_loss_risk(row):
    return min(1, (row["Viscosity_cP"] / 120) * (row["Solid_Content_%"] / 20))

def emulsion_risk(row):
    return min(1, (100 - row["Oil_Water_Ratio"]) / 100 + (100 - row["Emulsion_Stability"]) / 100)

def reactivity_score(row):
    score = 0
    if row["Shale_Reactiveness"] == "High":
        score += 1
    elif row["Shale_Reactiveness"] == "Medium":
        score += 0.5
    return score

# پارامترها
num_wells = 10
num_rows_per_well = 15_552_000
chunk_size = 1_000_000
output_dir = "synthetic_fdms_chunks"
os.makedirs(output_dir, exist_ok=True)

bit_types = ["PDC", "Tricone", "Diamond"]
formation_types = ["Sandstone", "Limestone", "Shale", "Dolomite"]
shale_reactivity = ["Low", "Medium", "High"]

well_info = [{
    "WELL_ID": f"WELL_{i+1}",
    "LAT": 28.0 + i * 0.01,
    "LONG": 52.0 + i * 0.01
} for i in range(num_wells)]

for i, info in enumerate(well_info):
    print(f"\n🚀 شروع تولید داده برای {info['WELL_ID']}")
    filepath = os.path.join(output_dir, f"FDMS_well_{info['WELL_ID']}.parquet")
    if os.path.exists(filepath):
        os.remove(filepath)

    shift = i * 0.1
    scale = 1 + (i % 5) * 0.05

    for start in range(0, num_rows_per_well, chunk_size):
        end = min(start + chunk_size, num_rows_per_well)
        size = end - start

        df = pd.DataFrame({
            "Depth_m": np.random.normal(3000 + shift*500, 800 * scale, size).clip(1000, 6000),
            "ROP_mph": np.random.normal(20 + shift*2, 8 * scale, size).clip(5, 50),
            "WOB_kgf": np.random.normal(15000 + shift*1000, 5000 * scale, size).clip(5000, 30000),
            "Torque_Nm": np.random.normal(1000 + shift*50, 400 * scale, size).clip(200, 2000),
            "Pump_Pressure_psi": np.random.normal(3000 + shift*500, 1000 * scale, size).clip(1000, 6000),
            "Mud_FlowRate_LPM": np.random.uniform(100 + shift*20, 800 + shift*30, size),
            "MWD_Vibration_g": np.random.uniform(0.1, 3.0 + shift, size),
            "Bit_Type": np.random.choice(bit_types, size),
            "Mud_Weight_ppg": np.random.normal(11 + shift, 1.5 * scale, size).clip(8.5, 15),
            "Viscosity_cP": np.random.normal(70 + shift*5, 20 * scale, size).clip(30, 120),
            "Plastic_Viscosity": np.random.normal(30 + shift*2, 8 * scale, size).clip(10, 50),
            "Yield_Point": np.random.normal(20 + shift*2, 6 * scale, size).clip(5, 40),
            "pH_Level": np.random.normal(8.5, 1.2 * scale, size).clip(6.5, 11),
            "Solid_Content_%": np.random.uniform(1, 20, size),
            "Chloride_Concentration_mgL": np.random.normal(50000 + shift*5000, 20000 * scale, size).clip(100, 150000),
            "Oil_Water_Ratio": np.random.uniform(10, 90, size),
            "Emulsion_Stability": np.random.uniform(30, 100, size),
            "Formation_Type": np.random.choice(formation_types, size),
            "Pore_Pressure_psi": np.random.normal(8000 + shift*500, 2000 * scale, size).clip(3000, 15000),
            "Fracture_Gradient_ppg": np.random.normal(15 + shift*0.2, 1.5 * scale, size).clip(13, 18),
            "Stress_Tensor_MPa": np.random.normal(40 + shift*2, 15 * scale, size).clip(10, 80),
            "Young_Modulus_GPa": np.random.normal(30 + shift*3, 10 * scale, size).clip(5, 70),
            "Poisson_Ratio": np.random.uniform(0.2, 0.35, size),
            "Brittleness_Index": np.random.uniform(0, 1, size),
            "Shale_Reactiveness": np.random.choice(shale_reactivity, size),
        })

        df["Fluid_Loss_Risk"] = df.apply(fluid_loss_risk, axis=1)
        df["Emulsion_Risk"] = df.apply(emulsion_risk, axis=1)
        df["Rock_Fluid_Reactivity"] = df.apply(reactivity_score, axis=1)
        df["Formation_Damage_Index"] = (
            df["Emulsion_Risk"] * 0.4 +
            df["Rock_Fluid_Reactivity"] * 0.3 +
            df["Brittleness_Index"] * 0.2 +
            np.random.normal(0, 0.05, size)
        )

        df["WELL_ID"] = info["WELL_ID"]
        df["LAT"] = info["LAT"]
        df["LONG"] = info["LONG"]
        df["timestamp"] = pd.to_datetime('2023-01-01 00:00:00') + pd.to_timedelta(start + df.index, unit='s')

        df.to_parquet(
            filepath,
            index=False,
            engine='fastparquet',  # ✅ تغییر مهم
            compression='snappy',
            append=os.path.exists(filepath)
        )

        print(f"✅ {info['WELL_ID']} | chunk {start:,} تا {end:,} ذخیره شد.")

print("\n🎉 تولید و ذخیره‌سازی داده‌ها با موفقیت انجام شد!")
