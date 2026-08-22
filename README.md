# EV Charging Energy Consumption Prediction

## Project Overview

This project uses Machine Learning to predict the energy consumed during an Electric Vehicle (EV) charging session.

The project uses historical EV charging session data containing information such as vehicle model, battery capacity, charging rate, charging time, state of charge, temperature, charger type, and user type.

## Problem Statement

EV charging station operators need to estimate the energy required by a charging session to improve station planning and electricity load management.

This project applies regression-based Machine Learning models to predict:

- Energy Consumed (kWh)
- Charging Duration (hours) - planned future extension

## Dataset
This project uses the Electric Vehicle Charging Patterns dataset from Kaggle.

Dataset Source:
https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns

License: Apache License 2.0

The dataset is used for educational and Machine Learning purposes.

Dataset: Electric Vehicle Charging Patterns

Dataset contains 1,320 charging sessions and 20 features.

Important features include:

- Vehicle Model
- Battery Capacity (kWh)
- Charging Rate (kW)
- Charging Start Time
- Charging End Time
- State of Charge (Start %)
- State of Charge (End %)
- Distance Driven
- Temperature
- Vehicle Age
- Charger Type
- User Type

Target variable:

- Energy Consumed (kWh)

## Machine Learning Models

The project compares different regression algorithms:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Gradient Boosting Regression

The models are evaluated using:

- MAE
- RMSE
- R² Score

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
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── model_comparison.py
│   ├── data_analysis.py
│   └── predict.py
│
├── main.py
├── requirements.txt
└── README.md