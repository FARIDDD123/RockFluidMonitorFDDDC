# Project Dataset Access

## 📁 Dataset Files
| File Name | Type | Size | Access Link | Viewing Tools |
|----------|-----|-----|-------------|------------------|
| `drilling_sensor_data_1m.csv` | CSV | 96MB | [Download from Google Drive](https://drive.google.com/file/d/1wvTqjpmhjLuOepNzxOX9VyNMOxwEFt0B/view?usp=drive_link) | Excel, LibreOffice, Notepad++ |
| `merged_10m_dataset.parquet` | Parquet | 663MB | [Download from Google Drive](https://drive.google.com/file/d/1--OvgMkTGYsZOGbGarp_PANcyDGDHfHw/view?usp=drive_link) | Online tools or Python/Pandas |
| `balanced_drilling_dataset.parquet` | Parquet | 1.77GB | [Download from Google Drive](https://drive.google.com/file/d/1IOtWthS4YkGoTj-ytF1ZsgGIlMeJof_R/view?usp=drive_link) | Online tools or Python/Pandas |
| `ROP_data.csv` | CSV | 614KB | [Download from Google Drive](https://drive.google.com/file/d/1LSyvWoXo1YBWiazjmwMF7NOU4ajRmg7k/view?usp=drive_link) | Excel, LibreOffice, Notepad++ |
| `16A(78)-32_time_data_10s_intervals_simplified.csv` | CSV | 52.2MB | [Download from Google Drive](https://drive.google.com/file/d/1ooxMee6LruQNcCCvyKz3gg0kIV0-2EI-/view?usp=drive_link) | Excel, LibreOffice, Notepad++ |
| `DRILL_Tripping_Circulation_Data_All.parquet` | Parquet | 19.6MB | [Download from Google Drive](https://drive.google.com/file/d/1ZPLPAeUEf5gHJ6s5h8ieZaQHWROhxR0t/view?usp=drive_link) | Online tools or Python/Pandas |
| `LWD_logs.parquet` | Parquet | 109MB | [Download from Google Drive](https://drive.google.com/file/d/1ITLy2wHAz-ZewVQamrwj2MSJ3iCxrgK3/view?usp=drive_link) | Online tools or Python/Pandas |
| `LWD_DRILL_data.parquet` | Parquet | 4.3MB | [Download from Google Drive](https://drive.google.com/file/d/1S5bogaZQeLuHdM13U-hhnyC60Cf5mHIC/view?usp=drive_link) | Online tools or Python/Pandas |
| `DRILL_LWD_MWD_Data.parquet` | Parquet | 3MB | [Download from Google Drive](https://drive.google.com/file/d/13TO0IDGZ6PWi4eyMpn3dPDmfQXHTcD6l/view?usp=drive_link) | Online tools or Python/Pandas |
| `DRILL_LWD_Sensor_Data.parquet` | Parquet | 3MB | [Download from Google Drive](https://drive.google.com/file/d/1V8LKcmAMDApcP1FjBjKi96gO_UsQ3fje/view?usp=drive_link) | Online tools or Python/Pandas |
| `DrillMech__LWD/MWD_Time_Based_R02_180-2221m_MD.csv` | CSV | 12MB | [Download from Google Drive](https://drive.google.com/file/d/1UDMVufdhoOvwWAeH7oHqDKaza5oX8X7Q/view?usp=drive_link) | Excel, LibreOffice, Notepad++ |
| `DrillingMechanics_LWD/MWD_Depth_Based_146-3285m_MD.csv` | CSV | 4.3MB | [Download from Google Drive](https://drive.google.com/file/d/1v9hVIzINqVMrqo4zN29f6G26jMqG9SjN/view?usp=drive_link) | Excel, LibreOffice, Notepad++ |

## 🛠 How to Use the Datasets

### For CSV File:
1. Click the Google Drive link
2. Select the "Download" button
3. Open with:
   - Microsoft Excel
   - LibreOffice Calc (on Ubuntu: sudo apt install libreoffice-calc)
   - Any text editor

## Professional Method for parquet dataset(Using Python):
```python
import pandas as pd
df = pd.read_parquet('merged_10m_dataset.parquet')
print(df.head())
```

## 🏗️ What is Parquet Format?
**Apache Parquet** is an advanced **column-oriented** file format with superior compression capabilities, specifically designed for big data processing. Unlike row-based CSV, Parquet stores data in a columnar fashion.

### 🔥 Key Advantages: Parquet vs CSV

| Feature          | Parquet                          | CSV                     |
|----------------|----------------------------------|-------------------------|
| **Structure**   | Columnar (optimized for queries) | Row-based               |
| **File Size**   | 75% smaller                      | Large                   |
| **Read Speed**  | Up to 10x faster                 | Slow                    |
| **Data Types**  | Preserves types (Timestamp, Decimal)| Raw text only         |
| **Schema**      | Fixed schema definition          | No schema               |
| **Compression** | Advanced (Snappy, Gzip)          | Typically uncompressed  |

### 💡 Why Parquet was chosen for this project?
1. **Faster processing** of high-volume drilling data
2. **Reduced storage costs** (especially with 600MB+ datasets)
3. **Selective column reading** without full file loading
4. **Compatibility** with Big Data ecosystems (Hadoop, Spark, Pandas)

```python
# Reading specific columns only
df = pd.read_parquet(
    'merged_10m_dataset.parquet',
    columns=['depth', 'pressure', 'temperature']  # RAM optimization
)

# Conversion to CSV if needed
df.to_csv('output.csv', index=False)
