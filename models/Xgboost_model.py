import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
from xgboost import XGBClassifier
import gc
from sklearn.preprocessing import MinMaxScaler
# === Load and select columns ===
df = pd.read_parquet('\\Users\\alireza\\Desktop\\alireza\\fddc\\synthetic_fdms_chunks\\FDMS_well_WELL_1.parquet', engine='fastparquet')
col = ['Oil_Water_Ratio', 'Emulsion_Stability', 'Formation_Damage_Index', 'Emulsion_Risk', 'timestamp']
df = df[col]
# === Binary Target ===
def make_binary(df, threshold=0.5):
    df['Emulsion_Risk_binary'] = np.where(df['Emulsion_Risk'] > threshold, 1, 0)
    return df.drop(columns=['Emulsion_Risk_binary']), df['Emulsion_Risk_binary']

# === Feature Engineering ===
def future_eng(df):
    df.set_index(['timestamp'], inplace=True)
    er = df['Emulsion_Risk']
    df.drop(columns=['Emulsion_Risk'], inplace=True)
    for col1 in df.columns:
        df[f'{col1}_dy_dx'] = np.gradient(df[col1])  
        df[f'{col1}_std_10s'] = df[f'{col1}'].groupby(pd.Grouper(freq='2s')).std()
        df[f'{col1}_mean_10s'] = df[f'{col1}'].groupby(pd.Grouper(freq='2s')).mean()
        df[f'{col1}_tanh'] = np.tanh(df[col1])
        df[f'{col1}/Emulsion_Risk_shift_1s'] = df[col1] / er.shift(1)
        df[f'{col1}_ma'] = df[col1].rolling(window=20).mean()
        df[f'{col1}_ema'] = df[col1].ewm(span=20, adjust=False).mean()
        df[f"{col1}_diff"] = df[col1].diff()
        df[f"{col1}_to_ES_ratio"] = df[col1].shift(1) / (df["Emulsion_Stability"].shift(1) + 1e-6)
    df["OWR_rolling_std_10"] = df["Oil_Water_Ratio"].rolling(window=10).std()
    df['Emulsion_Risk_shift_1s'] = er.shift(1)
    return df

# === Preprocessing ===
x, y = make_binary(df)
X = future_eng(x)
X.reset_index(inplace=True)
X.drop(columns=['timestamp'], inplace=True)
X.fillna(0, inplace=True)
gc.collect()

# === Train/Test Split ===
tss = TimeSeriesSplit(n_splits=6)

# === Model Training ===
model = XGBClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.01,
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0
)
for fold, (train_index, test_index) in enumerate(tss.split(X)):
        print(f'Fold {fold}')
        x_train_raw, y_train = X.iloc[train_index], y.iloc[train_index]
        x_test_raw, y_test = X.iloc[test_index], y.iloc[test_index]

        # === MinMax Scaling (fit only on training data)
        scaler = MinMaxScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train_raw), columns=x_train_raw.columns, index=x_train_raw.index)
        x_test = pd.DataFrame(scaler.transform(x_test_raw), columns=x_test_raw.columns, index=x_test_raw.index)

        # === Fit and Evaluate ===
        model.fit(x_train, y_train)
        y_pred_test = model.predict(x_test)
        y_proba_test = model.predict_proba(x_test)[:, 1]

        print(f"classification_report:\n{classification_report(y_true=y_test, y_pred=y_pred_test, zero_division=0)}")
# === Inference on WELL_2 ===
df2 = pd.read_parquet('\\Users\\alireza\\Desktop\\alireza\\fddc\\synthetic_fdms_chunks\\FDMS_well_WELL_2.parquet', engine='fastparquet').iloc[:10_000]
df2 = df2[col]
X2, y2 = make_binary(df2)
X2 = future_eng(X2)
X2.reset_index(inplace=True)
X2.drop(columns=['timestamp'], inplace=True)
X2.fillna(0, inplace=True)

y_pred_test2 = model.predict(X2)
y_proba_test2 = model.predict_proba(X2)[:, 1]

print(f"\nclassification_report on WELL_2:\n{classification_report(y_true=y2, y_pred=y_pred_test2, zero_division=0)}")

# === ROC and Confusion Matrix ===
roc_auc = roc_auc_score(y2, y_proba_test2)
cm = confusion_matrix(y2, y_pred_test2)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fpr, tpr, _ = roc_curve(y2, y_proba_test2)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()



