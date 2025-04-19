import numpy as np
import pandas as pd
from tqdm import tqdm

# تعداد نمونه‌ها
n_samples = 10_000_000

# تابع برای تولید داده با نویز کنترل‌شده
def noisy_normal(mean, std, size):
    return np.random.normal(loc=mean, scale=std, size=size)

# تولید داده‌ها
data = pd.DataFrame({
    "pressure": noisy_normal(5000, 300, n_samples),         # psi
    "flow_rate": noisy_normal(600, 50, n_samples),          # gpm
    "viscosity": noisy_normal(35, 5, n_samples),            # cp
    "depth": noisy_normal(12000, 500, n_samples),           # ft
    "temperature": noisy_normal(180, 15, n_samples),        # F
    "shear_rate": noisy_normal(100, 20, n_samples),         # s^-1
    "salinity": noisy_normal(3.5, 0.5, n_samples),          # wt%
    "ph": noisy_normal(7.0, 0.5, n_samples),                # unitless
    "mud_weight": noisy_normal(10.5, 0.3, n_samples),       # ppg
    "rock_type": np.random.choice(["shale", "sandstone", "limestone"], n_samples),
})

# برچسب‌گذاری برای مدل‌های ML (مثلاً تلفات سیال بالا/پایین)
# قانون ساده: اگر فشار بالا + ویسکوزیته پایین + عمق زیاد → ریسک بالا
data["fluid_loss_risk"] = (
    (data["pressure"] > 5200) &
    (data["viscosity"] < 33) &
    (data["depth"] > 12500)
).astype(int)

# ذخیره داده در فایل CSV
data.to_csv("synthetic_drilling_data.csv", index=False)

print("✅ 10 میلیون داده تولید و ذخیره شد.")
