# Real-time Fluid Loss Risk Prediction System

This project implements a real-time prediction service for drilling fluid loss risk using XGBoost and LSTM models.

## Features
- Real-time data processing with Kafka
- Dual-model prediction (XGBoost + LSTM)
- Alert system with <2s latency

## Requirements
- Python 3.8+
- Kafka server
- See requirements.txt for Python dependencies

## Usage
1. Train models: Run all cells in the notebook
2. Start Kafka server
3. Run the producer/consumer cells