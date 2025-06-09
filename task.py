#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('pip', 'install fastparquet')




# In[4]:


import pandas as pd
import numpy as np
import os

# تابع‌های تولید شاخص‌ها
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

# تابع افزودن نویز گوسی
def add_noise(df, columns, noise_level=0.05):
    for col in columns:
        noise = np.random.normal(0, noise_level * df[col].std(), len(df))
        df[col] += noise
    return df

# تابع افزودن داده‌های گمشده به‌صورت تصادفی
def add_missing_data(df, missing_rate=0.03):
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            idx = np.random.choice(df.index, size=int(missing_rate * len(df)), replace=False)
            df.loc[idx, col] = np.nan
    return df

# اجرای اصلی
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

        # 👇 وابستگی بین ویژگی‌ها
        depth = np.random.normal(3000 + shift*500, 800 * scale, size).clip(1000, 6000)
        mud_weight = np.random.normal(11 + shift, 1.5 * scale, size).clip(8.5, 15)
        viscosity = np.random.normal(70 + shift*5, 20 * scale, size).clip(30, 120)

        df = pd.DataFrame({
            "Depth_m": depth,
            "ROP_mph": np.random.normal(20 + shift*2, 8 * scale, size).clip(5, 50),
            "WOB_kgf": np.random.normal(15000 + shift*1000, 5000 * scale, size).clip(5000, 30000),
            "Torque_Nm": np.random.normal(1000 + shift*50, 400 * scale, size).clip(200, 2000),
            "Pump_Pressure_psi": 500 + mud_weight * 180 + np.random.normal(0, 300, size),  # ← رابطه با mud_weight
            "Mud_FlowRate_LPM": 10 + (depth / 10) + np.random.normal(0, 100, size),       # ← رابطه با depth
            "MWD_Vibration_g": np.random.uniform(0.1, 3.0 + shift, size),
            "Bit_Type": np.random.choice(bit_types, size),
            "Mud_Weight_ppg": mud_weight,
            "Viscosity_cP": viscosity,
            "Plastic_Viscosity": viscosity * 0.4 + np.random.normal(0, 5, size),
            "Yield_Point": viscosity * 0.2 + np.random.normal(0, 3, size),
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

        # 🧠 روابط، نویز و داده گمشده
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = add_noise(df, numeric_cols, noise_level=0.05)
        df = add_missing_data(df, missing_rate=0.03)

        # 🔬 محاسبه شاخص‌ها
        df["Fluid_Loss_Risk"] = df.apply(fluid_loss_risk, axis=1)
        df["Emulsion_Risk"] = df.apply(emulsion_risk, axis=1)
        df["Rock_Fluid_Reactivity"] = df.apply(reactivity_score, axis=1)
        df["Formation_Damage_Index"] = (
            df["Emulsion_Risk"] * 0.4 +
            df["Rock_Fluid_Reactivity"] * 0.3 +
            df["Brittleness_Index"] * 0.2 +
            np.random.normal(0, 0.05, size)
        )

        # افزودن اطلاعات چاه و زمان
        df["WELL_ID"] = info["WELL_ID"]
        df["LAT"] = info["LAT"]
        df["LONG"] = info["LONG"]
        df["timestamp"] = pd.to_datetime('2023-01-01 00:00:00') + pd.to_timedelta(start + df.index, unit='s')

        # ذخیره‌سازی
        df.to_parquet(
            filepath,
            index=False,
            engine='fastparquet',
            compression='snappy',
            append=os.path.exists(filepath)
        )

        print(f"✅ {info['WELL_ID']} | chunk {start:,} تا {end:,} ذخیره شد.")

print("\n🎉 تولید و ذخیره‌سازی داده‌ها با موفقیت انجام شد!")


# In[8]:


import os

output_dir = "synthetic_fdms_chunks"
files = os.listdir(output_dir)
parquet_files = [f for f in files if f.endswith(".parquet")]
print("فایل‌های Parquet موجود:", parquet_files)


# In[7]:


get_ipython().system('pip install pyarrow')


# In[8]:


import pyarrow.parquet as pq
import pandas as pd

file_path = "synthetic_fdms_chunks/FDMS_well_WELL_1.parquet"
batch_size = 100_000  # یا 1_000_000 برای سیستم قوی‌تر

# خواندن فقط بخشی از فایل
parquet_file = pq.ParquetFile(file_path)
batch = parquet_file.read_row_group(0).to_pandas()

# یا فقط ستون‌های خاص را بخوان
# batch = parquet_file.read_row_group(0, columns=['Formation_Damage_Index', ...]).to_pandas()

print("✅ تعداد ردیف‌ها:", len(batch))


# In[9]:


import pyarrow.parquet as pq
import pandas as pd
import os

# مسیر فایل نمونه (می‌تونی تغییر بدی به هر چاه دیگر)
file_path = "synthetic_fdms_chunks/FDMS_well_WELL_1.parquet"
output_file = "FDMS_well_WELL_1_sample_processed.parquet"

# تابع تعیین practicality
def assess_practicality(row):
    if row["Formation_Damage_Index"] >= 0.75:
        return "Non-Practical"
    if row["Fluid_Loss_Risk"] > 0.85:
        return "Non-Practical"
    if row["Rock_Fluid_Reactivity"] == 1 and row["Emulsion_Risk"] > 0.65:
        return "Non-Practical"
    return "Practical"

# خواندن فقط 100,000 ردیف اول از فایل پارکت با pyarrow
parquet_file = pq.ParquetFile(file_path)
batch = parquet_file.read_row_group(0).to_pandas().head(100_000)

# اضافه کردن practicality
batch["damage_practicality"] = batch.apply(assess_practicality, axis=1)

# نمایش آمار اولیه
print("📊 آمار:")
print(batch["damage_practicality"].value_counts())

# ذخیره در فایل جدید
batch.to_parquet(output_file, index=False, engine="fastparquet", compression="snappy")
print(f"\n✅ ذخیره فایل: {output_file}")


# In[10]:


import pyarrow.parquet as pq
import pandas as pd
import os

# تابع practicality
def assess_practicality(row):
    if row["Formation_Damage_Index"] >= 0.75:
        return "Non-Practical"
    if row["Fluid_Loss_Risk"] > 0.85:
        return "Non-Practical"
    if row["Rock_Fluid_Reactivity"] == 1 and row["Emulsion_Risk"] > 0.65:
        return "Non-Practical"
    return "Practical"

# چاه‌هایی که نمونه‌گیری می‌کنیم
wells = ["WELL_1", "WELL_5", "WELL_10"]
results = {}

for well in wells:
    file_path = f"synthetic_fdms_chunks/FDMS_well_{well}.parquet"
    parquet_file = pq.ParquetFile(file_path)
    
    print(f"\n📥 در حال نمونه‌گیری از {well}")
    df = parquet_file.read_row_group(0).to_pandas().head(100_000)
    
    df["damage_practicality"] = df.apply(assess_practicality, axis=1)
    stats = df["damage_practicality"].value_counts()
    
    # ذخیره آمار
    results[well] = stats.to_dict()
    print(stats)

# نمایش جدول مقایسه‌ای
print("\n📊 مقایسه نهایی:")
summary = pd.DataFrame(results).T
summary["% Practical"] = (summary["Practical"] / summary.sum(axis=1) * 100).round(2)
summary["% Non-Practical"] = (summary["Non-Practical"] / summary.sum(axis=1) * 100).round(2)
print(summary)


# In[2]:


get_ipython().run_line_magic('pip', 'install  fpdf')


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# 📥 بارگذاری داده نمونه
df = pd.read_parquet("FDMS_well_WELL_1_sample_processed.parquet")

# 🧮 آمار کلی
stats = df["damage_practicality"].value_counts()
total = stats.sum()
percentages = (stats / total * 100).round(2)

# 📊 نمودار دایره‌ای
plt.figure(figsize=(5, 5))
plt.pie(stats, labels=stats.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
plt.title("Damage Practicality Distribution")
plt.tight_layout()
plt.savefig("pie_chart.png")
plt.close()

# 📈 هیستوگرام Formation Damage Index
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Formation_Damage_Index", hue="damage_practicality", bins=30, kde=True)
plt.title("Formation Damage Index Distribution")
plt.xlabel("FDI")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("histogram.png")
plt.close()

# 📄 ایجاد PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "Formation Damage Practicality Report", ln=True)

# ➤ بخش اول: منطق
# ➤ آمار
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "Summary Statistics", ln=True)
pdf.set_font("Arial", '', 12)
for cat in stats.index:
    pdf.cell(0, 10, f"{cat}: {stats[cat]:,} ({percentages[cat]}%)", ln=True)

# ➤ تصاویر
pdf.image("pie_chart.png", w=100)
pdf.image("histogram.png", w=170)

# ➤ جداول نمونه
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "Sample Records", ln=True)

pdf.set_font("Arial", '', 10)

# ۵ رکورد Practical
practical_sample = df[df["damage_practicality"] == "Practical"].head(5)
nonpractical_sample = df[df["damage_practicality"] == "Non-Practical"].head(5)

pdf.cell(0, 8, "Practical Samples:", ln=True)
for idx, row in practical_sample.iterrows():
    pdf.cell(0, 8, f"FDI={row['Formation_Damage_Index']:.2f}, FluidLoss={row['Fluid_Loss_Risk']:.2f}, Emulsion={row['Emulsion_Risk']:.2f}", ln=True)

pdf.cell(0, 8, "Non-Practical Samples:", ln=True)
for idx, row in nonpractical_sample.iterrows():
    pdf.cell(0, 8, f"FDI={row['Formation_Damage_Index']:.2f}, FluidLoss={row['Fluid_Loss_Risk']:.2f}, Emulsion={row['Emulsion_Risk']:.2f}", ln=True)

# 📤 ذخیره PDF
pdf.output("practicality_report.pdf")
print("✅ گزارش practicality_report.pdf با موفقیت ساخته شد.")


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# بارگذاری داده پردازش‌شده
df = pd.read_parquet("FDMS_well_WELL_1_sample_processed.parquet")
stats = df["damage_practicality"].value_counts()
percentages = (stats / stats.sum() * 100).round(2)

# نمودار دایره‌ای
plt.figure(figsize=(5, 5))
plt.pie(stats, labels=stats.index, autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
plt.title("Damage Practicality Distribution")
plt.savefig("simple_pie_chart.png")
plt.close()

# هیستوگرام
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Formation_Damage_Index", hue="damage_practicality", bins=30, kde=True)
plt.title("Formation Damage Index Distribution")
plt.savefig("simple_histogram.png")
plt.close()

# ساخت PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Formation Damage Report", ln=True)

pdf.set_font("Arial", '', 11)
pdf.multi_cell(0, 8, """
Damage severity is categorized as follows:

- If Formation_Damage_Index ≥ 0.75 → Non-Practical
- If Fluid_Loss_Risk > 0.85 → Non-Practical
- If Rock_Fluid_Reactivity = 1 and Emulsion_Risk > 0.65 → Non-Practical
- Otherwise → Practical
""")

# آمار
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "Summary:", ln=True)
pdf.set_font("Arial", '', 11)
for cat in stats.index:
    pdf.cell(0, 8, f"{cat}: {stats[cat]:,} ({percentages[cat]}%)", ln=True)

# نمودارها
pdf.image("simple_pie_chart.png", w=100)
pdf.image("simple_histogram.png", w=180)

# نمونه رکوردها
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "Sample Records:", ln=True)
pdf.set_font("Arial", '', 10)

pdf.cell(0, 8, "Practical:", ln=True)
for _, row in df[df["damage_practicality"] == "Practical"].head(5).iterrows():
    line = f"FDI={row['Formation_Damage_Index']:.2f} | Fluid={row['Fluid_Loss_Risk']:.2f} | Emul={row['Emulsion_Risk']:.2f}"
    pdf.cell(0, 8, line, ln=True)

pdf.cell(0, 8, "Practical:", ln=True)
for _, row in df[df["damage_practicality"] == "Practical"].head(5).iterrows():
    line = f"FDI={row['Formation_Damage_Index']:.2f} | Fluid={row['Fluid_Loss_Risk']:.2f} | Emul={row['Emulsion_Risk']:.2f}"
    pdf.cell(0, 8, line, ln=True)

pdf.cell(0, 8, "Non-Practical:", ln=True)
for _, row in df[df["damage_practicality"] == "Non-Practical"].head(5).iterrows():
    line = f"FDI={row['Formation_Damage_Index']:.2f} | Fluid={row['Fluid_Loss_Risk']:.2f} | Emul={row['Emulsion_Risk']:.2f}"
    pdf.cell(0, 8, line, ln=True)



