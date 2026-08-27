import pandas as pd

DATA_PATH = "data/ev_workplace_charging_data.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("EV WORKPLACE CHARGING DATASET")
print("=" * 50)

# ------------------------------------------------
# 1. Dataset shape
# ------------------------------------------------

print("\nDataset Shape:")
print(df.shape)


# ------------------------------------------------
# 2. Columns
# ------------------------------------------------

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------
# 3. Missing values
# ------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------------
# 4. Energy statistics
# ------------------------------------------------

print("\nEnergy Consumption Statistics:")
print(df["kwhTotal"].describe())


# ------------------------------------------------
# 5. Charging duration statistics
# ------------------------------------------------

print("\nCharging Duration Statistics:")
print(df["chargeTimeHrs"].describe())


# ------------------------------------------------
# 6. Numerical correlations with energy
# ------------------------------------------------

print("\nCorrelation with Energy Consumption:")

numeric_df = df.select_dtypes(include="number")

energy_corr = (
    numeric_df.corr()["kwhTotal"]
    .sort_values(ascending=False)
)

print(energy_corr)


# ------------------------------------------------
# 7. Numerical correlations with duration
# ------------------------------------------------

print("\nCorrelation with Charging Duration:")

duration_corr = (
    numeric_df.corr()["chargeTimeHrs"]
    .sort_values(ascending=False)
)

print(duration_corr)


# ------------------------------------------------
# 8. Sample records
# ------------------------------------------------

print("\nSample Charging Sessions:")

sample_columns = [
    "kwhTotal",
    "chargeTimeHrs",
    "distance",
    "stationId",
    "locationId",
    "totalSessions"
]

# Only use columns that actually exist
available_columns = [
    col for col in sample_columns
    if col in df.columns
]

print(df[available_columns].head(10))