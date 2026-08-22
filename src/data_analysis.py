import pandas as pd

DATA_PATH = "C:/Users/mysel/OneDrive/Desktop/ev_charging prediction/data/ev_charging_patterns.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nEnergy Consumption Statistics:")
print(df["Energy Consumed (kWh)"].describe())

print("\nCorrelation with Energy Consumption:")

numerical_columns = [
    "Battery Capacity (kWh)",
    "Charging Rate (kW)",
    "State of Charge (Start %)",
    "State of Charge (End %)",
    "Distance Driven (since last charge) (km)",
    "Temperature (°C)",
    "Vehicle Age (years)",
    "Energy Consumed (kWh)"
]

correlation = df[numerical_columns].corr()

print(
    correlation["Energy Consumed (kWh)"]
    .sort_values(ascending=False)
)
print("\nCharging Duration Correlation:")

print(
    df[
        [
            "Energy Consumed (kWh)",
            "Charging Duration (hours)",
            "Charging Rate (kW)"
        ]
    ].corr()["Energy Consumed (kWh)"]
)
df["Calculated Energy"] = (
    df["Charging Duration (hours)"]
    * df["Charging Rate (kW)"]
)

print("\nActual vs Calculated Energy:")

print(
    df[
        [
            "Energy Consumed (kWh)",
            "Calculated Energy"
        ]
    ].head(10)
)