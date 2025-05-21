# 📄 مستندات کد پیش‌پردازش داده‌های حفاری

این اسکریپت پایتون برای **پیش‌پردازش داده‌های چاه‌های حفاری** طراحی شده و مراحل زیر را برای هر فایل `.parquet` انجام می‌دهد:

- بارگذاری داده
- حذف مقادیر گمشده (NaN)
- نرمال‌سازی ویژگی‌های عددی
- دسته‌بندی ویژگی‌های متنی
- شناسایی و حذف داده‌های پرت با Z-Score و IQR
- ذخیره داده‌های پاک‌شده، پرت‌ها و آمار اولیه (برای بازیابی)

---

## 📁 ساختار پوشه‌ها

```text
dataset/
├── fdms_well_datasets/       # ورودی فایل‌های .parquet خام برای هر چاه
├── processed/                # خروجی داده‌های پاک‌شده
├── outliers/                 # خروجی داده‌های پرت
├── means/                    # آمار میانگین و انحراف معیار هر چاه (JSON)
```

---

## 🔄 مراحل پردازش داده‌ها

### 1. ساخت مسیرهای خروجی
```python
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTLIER_DIR).mkdir(parents=True, exist_ok=True)
Path(MEAN_DIR).mkdir(parents=True, exist_ok=True)
```

### 2. خواندن فایل و حذف NaN
```python
df = pd.read_parquet(raw_path)
df = df.dropna()
```

### 3. ذخیره آماری ویژگی‌های عددی قبل از نرمال‌سازی
```python
original_means = df[numeric_cols].mean().to_dict()
original_stds = df[numeric_cols].std().to_dict()
```

### 4. نرمال‌سازی ویژگی‌های عددی
```python
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```

### 5. تبدیل ویژگی‌های متنی به عددی
```python
df = pd.get_dummies(df, columns=categorical_cols)
```

### 6. شناسایی داده‌های پرت

#### 🔹 Z-Score:
```python
z_scores = np.abs(stats.zscore(df[numeric_cols]))
outliers_z = (z_scores > 3).any(axis=1)
```

#### 🔹 IQR:
```python
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
```

#### 🔹 ترکیب نهایی:
```python
outliers = outliers_z | outliers_iqr
```

### 7. ذخیره داده‌های پرت و داده‌های تمیز
```python
df_outliers.to_parquet(outliers_path, index=False)
df_clean = df[~outliers]
df_clean.to_parquet(processed_path, index=False)
```

### 8. ذخیره آمار اولیه به فرمت JSON
```python
stats_to_save = {
    "WELL_ID": int(well_id),
    "mean": original_means,
    "std": original_stds
}
with open(mean_json_path, 'w') as f:
    json.dump(stats_to_save, f, indent=4)
```

---

## ✅ خروجی‌ها

| نوع داده              | مسیر خروجی                            |
|----------------------|----------------------------------------|
| داده پاک‌شده         | `dataset/processed/cleaned_<ID>.parquet` |
| داده‌های پرت         | `dataset/outliers/outliers_<ID>.parquet` |
| آمار اولیه (میانگین، std) | `dataset/means/mean_<ID>.json`             |
