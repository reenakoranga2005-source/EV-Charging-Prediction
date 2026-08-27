# EV Charging Energy Consumption Prediction

## Project Overview

This project uses Machine Learning to predict the energy consumed during an Electric Vehicle (EV) charging session.

The project uses historical EV charging session data containing information such as vehicle model, battery capacity, charging rate, charging time, state of charge, temperature, charger type, and user type.

## Problem Statement

EV charging station operators need to estimate the energy required by a charging session to improve station planning and electricity load management.

This project applies regression-based Machine Learning models to predict:

- **Energy Consumed (kWh)** (Primary Target)
- **Charging Duration (hours)** *(Planned future extension)*

## Dataset
This project utilizes the Electric Vehicle Charging Patterns dataset from Kaggle.

* **Source:** [Kaggle - Electric Vehicle Charging Patterns](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns)
* **License:** Apache License 2.0
* **Size:** 1,320 charging sessions across 20 features
* **Purpose:** Educational and Machine Learning research

### Key Features
* **Vehicle Metrics:** Vehicle Model, Battery Capacity (kWh), Vehicle Age
* **Session Metrics:** Charging Start/End Time, State of Charge (Start % & End %), Distance Driven
* **Environmental & Hardware:** Temperature, Charger Type, User Type
* **Target Variable:** `Energy Consumed (kWh)`

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