import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def report_missing(df):
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_report = missing_percent[missing_percent > 0].sort_values(ascending=False)
    print("\n📊 گزارش داده‌های گمشده:")
    print(missing_report)
    return missing_report

def encode_categoricals(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str).fillna("Missing")
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def process_datetime(df):
    df = df.copy()
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns
    for col in datetime_cols:
        df[col + "_year"] = df[col].dt.year
        df[col + "_month"] = df[col].dt.month
        df[col + "_day"] = df[col].dt.day
        df[col + "_hour"] = df[col].dt.hour
    df.drop(columns=datetime_cols, inplace=True)
    return df

def scale_numeric(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler

def get_models(is_classification):
    if is_classification:
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric="mlogloss", random_state=42),
            "KNN": KNeighborsClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
            "CatBoost": CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, verbose=0, random_seed=42)
        }
    else:
        return {
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, random_state=42),
            "KNN": KNeighborsRegressor(),
            "CatBoost": CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, verbose=0, random_seed=42)
        }

def evaluate_model(y_true, y_pred, is_classification, model=None, X_test=None):
    if is_classification:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro')
        }
        if model and hasattr(model, 'predict_proba') and X_test is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, model.predict_proba(X_test), multi_class='ovr')
            except:
                metrics['auc'] = None
    else:
        metrics = {
            'rmse': mean_squared_error(y_true, y_pred, squared=False),
            'mae': mean_absolute_error(y_true, y_pred)
        }
    return metrics

def preprocess_full(df):
    df = process_datetime(df)
    df = encode_categoricals(df)
    df, scaler = scale_numeric(df)
    return df, scaler

def impute_missing(df):
    results = []
    shap.initjs()
    df, scaler = preprocess_full(df)  # یکبار پیش‌پردازش کامل

    missing_cols = df.columns[df.isnull().any()]

    for col in missing_cols:
        print(f"\n🔧 پردازش ستون: {col}")
        df_train = df[df[col].notnull()].copy()
        df_test = df[df[col].isnull()].copy()

        if df_train.shape[0] < 500:
            print("⛔ داده کافی برای آموزش وجود ندارد.")
            continue

        y = df_train[col]
        X = df_train.drop(columns=[col])

        y_type = y.dtype
        is_classification = y_type == 'object' or y.nunique() <= 10

        if is_classification:
            y = LabelEncoder().fit_transform(y.astype(str))

        X_pred = df_test[X.columns].copy()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        best_model = None
        best_metrics = None
        best_model_name = None
        best_score = -np.inf if is_classification else np.inf

        for name, model in get_models(is_classification).items():
            try:
                if "XGBoost" in name and not is_classification:
                    model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)
                elif "XGBoost" in name and is_classification:
                    model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                metrics = evaluate_model(y_val, y_pred, is_classification, model, X_val)
                print(f"  📌 {name} | {metrics}")

                score = metrics['f1'] if is_classification else metrics['rmse']
                score_cmp = score > best_score if is_classification else score < best_score

                if score_cmp:
                    best_model = model
                    best_metrics = metrics
                    best_model_name = name
                    best_score = score

            except Exception as e:
                print(f"⚠️ خطا در مدل {name}: {e}")

        print(f"✅ بهترین مدل برای {col}: {best_model_name} | {best_metrics}")
        results.append((col, best_model_name, best_metrics))

        joblib.dump(best_model, os.path.join(OUTPUT_DIR, f"{col}_best_model10.pkl"))

        try:
            shap_input = X_train.select_dtypes(include=[np.number])
            explainer = shap.Explainer(best_model, shap_input)
            shap_values = explainer(shap_input[:200])
            plt.figure()
            shap.summary_plot(shap_values, shap_input[:200], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{col}_shap10.png"))
            plt.close()
        except Exception as e:
            print(f"⚠️ خطا در تحلیل SHAP برای {col}: {e}")

    return results

if __name__ == "__main__":
    print("📥 بارگذاری داده...")
    df = pd.read_parquet("synthetic_fdms_chunks/FDMS_well_WELL_10.parquet")
    df = df.sample(100_000, random_state=42)

    print("\n🚩 شروع فرآیند تحلیل داده‌های گمشده")
    missing_report = report_missing(df)
    evaluation_results = impute_missing(df)

    print("\n📊 نتایج نهایی مدل‌ها:")
    for col, model, metrics in evaluation_results:
        print(f"{col}: {model} -> {metrics}")
