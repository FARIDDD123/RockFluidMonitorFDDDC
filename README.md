# 🛢️ Software Requirements Specification (SRS)

## 💡 Formation Damage Monitoring System (FDMS)

---

## 1. Introduction

### 1.1 Purpose
The purpose of this document is to define the requirements for the intelligent FDMS system, which is responsible for real-time monitoring, 3D simulation, and prediction of formation damage during drilling and cementing operations. This system is developed using MWD/LWD data and machine learning.

### 1.2 System Scope
FDMS is designed as an integrated system comprising a data analysis engine, machine learning models, a 3D simulation engine, and an interactive management dashboard. The system is intended for real-time monitoring, predictive analysis, and operational decision optimization in drilling and cementing processes and is applicable to both onshore and offshore environments.

### 1.3 Definitions and Acronyms

| Term | Description |
|------|-------------|
| MWD | Measurement While Drilling |
| LWD | Logging While Drilling |
| ECD | Equivalent Circulating Density |
| OBM/WBM | Oil/Water Based Mud |
| SHAP | SHapley Additive exPlanations |
| FDMS | Formation Damage Monitoring System |

---

## 2. Overall Description

### 2.1 Vision
FDMS plays a critical role in preventing formation damage by providing real-time predictions and interactive simulations, enhancing operator decision-making.

### 2.2 Key Features
- ⏱️ Real-time processing of drilling data (MWD/LWD)
- 🤖 Intelligent prediction of fluid loss and emulsion risk
- 🖥️ 3D formation simulation
- 📊 Management dashboard with smart alerts

### 2.3 Constraints
- Requires high-quality drilling data
- Depends on GPU resources for model training
- Requires stable site network connectivity

---

## 3. Functional Requirements (FR)

### 3.1 Fluid Loss Risk Prediction
- [FR-1.1] Use of XGBoost model for risk prediction
- [FR-1.2] Feature importance analysis with SHAP
- [FR-1.3] Graphical display of results in dashboard

### 3.2 Emulsion Detection in Drilling
- [FR-2.1] Input time series: viscosity, temperature, shear rate
- [FR-2.2] Use LSTM model to detect emulsion formation
- [FR-2.3] Real-time alert upon detection

### 3.3 3D Formation Simulation
- [FR-3.1] Visualize stressed formation using Unity3D
- [FR-3.2] Enable operator interaction with the 3D model

### 3.4 Management Dashboard
- [FR-4.1] Display pressure, temperature, RPM, ECD
- [FR-4.2] Alert system for threshold violations
- [FR-4.3] ML-based operational recommendations

---

## 4. Non-Functional Requirements (NFR)

| Code | Requirement | Description |
|------|-------------|-------------|
| NFR-1 | Performance | Process drilling data at a minimum rate of 1Hz |
| NFR-2 | Reliability | Continue operating during temporary disconnections |
| NFR-3 | Scalability | Modular architecture for ML and UI components |
| NFR-4 | Security | Data exchange encryption and user authentication |
| NFR-5 | Maintainability | Full documentation and testability support |

---

## 5. Data Requirements

### 5.1 Suggested Data Structure (Sample for 10 wells)
```csv
well_id, lat, lon, datetime, depth, mud_type, rpm, spp, flow_rate, viscosity, temperature, ecd, shale_index, bit_type, lithology, cuttings_concentration, ph, cl_concentration, oil_water_ratio, emulsion_detected, lost_circulation, casing_pressure, annular_pressure
```

### 5.2 Data Sources
- Real field data from drilled wells
- Synthetic data generated via time-series algorithms (TimeGAN, SynthFlow)

---

## 6. High-Level Diagrams

### 6.1 System Architecture
```
[Data Acquisition] → [Processing Layer (Python/C++)]
                   → [ML Models: XGBoost, LSTM]
                   → [3D Simulator: Unity]
                   → [Backend API: FastAPI]
                   → [Frontend Dashboard: React + Plotly]
```

### 6.2 User Interaction Diagram
```
[User] ⇄ [Dashboard] ⇄ [Backend API] ⇄ [ML Engine / 3D Simulator / DB]
```

---

## 7. Appendices

- Initial Setup Guide:
  ```bash
  pip install -r requirements.txt
  uvicorn app.main:app --reload
  python train_fluid_loss_model.py
  ```

- Folder Structure:
  ```
  📁 data/ ← Drilling data
  📁 models/ ← Trained models
  📁 simulation/ ← Unity simulations
  📁 dashboard/ ← React frontend & API
  📁 notebooks/ ← Analyses and experiments
  ```

- Contact Information for Field Collaboration or Development:
  ```
  📧 email
  ☎️ phone number
  ```
