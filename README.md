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
