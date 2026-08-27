# EV Charging Energy Prediction - New Version

## Project Overview

This project implements machine learning models to forecast the total energy consumed during Electric Vehicle (EV) charging sessions. By leveraging historical operational data—including vehicle specs, battery parameters, ambient conditions, and session metrics—the system provides data-driven estimates to optimize charging infrastructure and load distribution.

## Problem Statement

EV charging station operators require accurate demand forecasting to balance electrical grid loads and plan capacity effectively. This repository utilizes supervised regression techniques to predict:

* **Energy Consumed (kWh)**: Primary target variable for energy demand estimation.
* **Charging Duration (hours)**: Secondary target for session planning *(planned extension)*.

## Dataset

This project utilizes the Electric Vehicle Charging Patterns dataset available on Kaggle.

* **Source**: [Kaggle - Electric Vehicle Charging Patterns](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns)
* **License**: Apache License 2.0
* **Volume**: 1,320 charging sessions across 20 attributes
* **Purpose**: Machine learning research and predictive modeling

### Feature Breakdown

* **Vehicle Metrics**: Vehicle Model, Battery Capacity (kWh), Vehicle Age
* **Session Metrics**: Charging Start/End Time, State of Charge (Start % & End %), Distance Driven
* **Environmental & Hardware**: Temperature, Charger Type, User Type
* **Target Variable**: `Energy Consumed (kWh)`

## Project Structure

```text
EV_Charging_Prediction/
│
├── data/
│   └── ev_charging.csv
│
├── models/
│   └── energy_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── data_analysis.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── model_comparison.py
│   └── predict.py
│
├── main.py
├── requirements.txt
└── README.md

## ⚠️ Project Limitations & Future Work

While this repository provides a baseline for predicting EV energy consumption, several operational and data constraints limit its direct application to live production grids.

### 1. Data Constraints & Scope
* **Sample Size & Overfitting:** The dataset is restricted to 1,320 charging sessions. This limits the use of deep learning models and increases the risk of overfitting on niche user behaviors.
* **Lack of Grid & Infrastructure Context:** The model lacks visibility into real-time grid constraints, transformer capacities, simultaneous station utilization, and local utility pricing signals.
* **Geographic & Seasonal Bias:** The data is pulled from a single static source and does not account for micro-climates, regional driving habits, or terrain variations.

### 2. Feature & Target Modeling Gaps
* **Mathematical Redundancy Risk:** In ideal scenarios, energy consumption is a deterministic formula: `Battery Capacity × (End % - Start %) / 100`. The machine learning models are highly reliant on these percentages, which may obscure real-world variables like charging cable efficiency losses or battery degradation.
* **Static Environment Metrics:** Ambient temperature is treated as a static value per session, failing to account for real-time weather drops or cabin pre-conditioning draws during the session.
* **Unimplemented Multi-Output Planning:** Charging Duration remains a "planned extension." Without forecasting *when* the load will clear, the system cannot perform comprehensive peak-demand scheduling.

### 3. Architectural & Deployment Limitations
* **Batch vs. Real-Time Inference:** The project stores a static `energy_model.pkl` file. It does not support online learning or real-time continuous updates as new EV models enter the market.
* **No API Layer:** The repository relies on localized execution scripts (`main.py`, `predict.py`). It lacks an API wrapper (such as FastAPI or Flask) required to integrate with live charging station management systems (CSMS) via standard protocols like OCPP.
* **No Automated Pipeline Orchestration:** There is no workflow management tool (e.g., Airflow, Prefect) to handle automated data re-ingestion, validation, and scheduled retraining.
