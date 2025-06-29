# 🛢️ Software Requirements Specification (SRS)

## 💡 Formation Damage Monitoring System (FDMS)

---

## 1. Introduction

### 1.1 Purpose
The purpose of this document is to define the requirements for the intelligent FDMS system, which is responsible for real-time monitoring and prediction of formation damage during drilling and cementing operations. This system leverages MWD/LWD data, predefined validation rules, and machine learning algorithms.

### 1.2 System Scope
FDMS is an integrated software platform consisting of a data validation pipeline, predictive maintenance engine, machine learning models, and a real-time dashboard for monitoring key parameters. It is applicable to both onshore and offshore drilling environments and is intended to minimize formation damage and optimize operational decisions.

### 1.3 Definitions and Acronyms

| Term         | Description                             |
|--------------|-----------------------------------------|
| MWD          | Measurement While Drilling              |
| LWD          | Logging While Drilling                  |
| ECD          | Equivalent Circulating Density          |
| OBM/WBM      | Oil/Water Based Mud                     |
| SHAP         | SHapley Additive exPlanations           |
| FDMS         | Formation Damage Monitoring System      |
| RMSE         | Root Mean Squared Error                 |
| MAE          | Mean Absolute Error                     |
| R²           | Coefficient of Determination            |

---

## 2. Overall Description

### 2.1 Vision
FDMS plays a critical role in drilling risk management by predicting formation damage and fluid-related anomalies in real time, enabling proactive and informed decision-making by engineers and operators.

### 2.2 Key Features
- ⏱️ Real-time data stream validation
- 🧠 Predictive maintenance using ML algorithms (XGBoost, LSTM, GRU)
- 🔍 SHAP-based feature importance analysis
- 📊 Interactive dashboard with alerting and analytics

### 2.3 Constraints
- Requires high-frequency, high-integrity sensor data
- Depends on reliable network access to perform real-time inference
- Needs GPU resources for model training and retraining

---

## 3. Functional Requirements (FR)

### 3.1 Data Validation Pipeline
- [FR-1.1] Enforce domain-specific rules (value range, null checks, unit consistency)
- [FR-1.2] Tag and log anomalies or corrupt records
- [FR-1.3] Route clean data to downstream ML modules

### 3.2 Fluid Loss and Emulsion Risk Detection
- [FR-2.1] Use XGBoost and regression for fluid loss forecasting
- [FR-2.2] Use LSTM and GRU models for detecting emulsion risk in time series data
- [FR-2.3] Trigger dashboard alerts when critical thresholds are predicted

### 3.3 Predictive Maintenance Engine
- [FR-3.1] Predict the likelihood of formation damage based on current drilling parameters
- [FR-3.2] Evaluate models using RMSE, MAE, and R² scores
- [FR-3.3] Compare different ML models and select best-performing one per well

### 3.4 Management Dashboard
- [FR-4.1] Display real-time operational parameters (pressure, temperature, ECD, RPM, etc.)
- [FR-4.2] Highlight validated vs. anomalous inputs
- [FR-4.3] Show ML predictions with feature attribution (via SHAP)

---

## 4. Non-Functional Requirements (NFR)

| Code   | Requirement    | Description                                        |
|--------|----------------|----------------------------------------------------|
| NFR-1  | Performance     | Process and validate sensor data at ≥1 Hz         |
| NFR-2  | Reliability     | Operate under temporary network loss conditions   |
| NFR-3  | Scalability     | Scalable to 50+ wells and multi-rig deployments   |
| NFR-4  | Security        | TLS encryption + RBAC for all users               |
| NFR-5  | Maintainability | Modular codebase with full documentation and CI/CD |

---

## 5. Data Requirements

### 5.1 Suggested Data Schema
```csv
well_id, lat, lon, datetime, depth, mud_type, rpm, spp, flow_rate, viscosity, temperature, ecd, shale_index, bit_type, lithology, cuttings_concentration, ph, cl_concentration, oil_water_ratio, emulsion_detected, lost_circulation, casing_pressure, annular_pressure
```

### 5.2 Data Sources
- Real-time MWD/LWD sensor data from active drilling wells
- Historical data from completed wells
- Synthetic time-series data generated using TimeGAN or SynthFlow for model augmentation

---

## 6. Machine Learning Details

### 6.1 Data Validation Logic
All incoming data is validated against predefined physical and logical rules:
- Acceptable range enforcement for all numeric features
- Type and unit conformity
- Automatic filling or removal of missing/invalid values
- Anomaly tagging for suspicious readings

### 6.2 Predictive Maintenance Models
The following models are used to predict and prevent formation damage:
- 🔢 **Regression Models**: Linear, Ridge, Polynomial
- 🌳 **XGBoost**: Gradient-boosted trees for tabular predictions
- 🔁 **LSTM & GRU**: Recurrent networks for sequential/temporal data

Each model is trained on labeled historical well data and evaluated using:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score (Goodness-of-fit)

Model selection is automated based on performance per site or lithology type.

---

## 7. High-Level System Diagrams

### 7.1 System Architecture
```
[Sensor Streams] → [Validation Layer]
                 → [ML Models: XGBoost, LSTM, GRU]
                 → [Backend API (FastAPI)]
                 → [Frontend Dashboard (React + Plotly)]
```

### 7.2 User Interaction Flow
```
[Operator] ⇄ [Dashboard] ⇄ [API Gateway] ⇄ [ML Engine / Logs / Database]
```

---

## 8. Appendices

### 8.1 Setup Instructions
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
python train_predictive_models.py
```

### 8.2 Folder Structure
```
📁 data/         ← Raw and validated drilling data  
📁 models/       ← Trained model artifacts (.pkl/.h5)  
📁 dashboard/    ← React + FastAPI backend  
📁 notebooks/    ← Data analysis & experiments  
📁 reports/      ← Evaluation metrics & logs  
```

### 8.3 Contact Information
```
📧  
☎️ 
```
