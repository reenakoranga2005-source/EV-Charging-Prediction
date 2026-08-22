import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------

DATA_PATH = "C:/Users/mysel/OneDrive/Desktop/ev_charging prediction/data/ev_charging_patterns.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# ------------------------------------------------
# 2. REMOVE DUPLICATES
# ------------------------------------------------

df = df.drop_duplicates()


# ------------------------------------------------
# 3. CREATE USEFUL FEATURES
# ------------------------------------------------

df["Charging Start Time"] = pd.to_datetime(
    df["Charging Start Time"]
)

# Extract hour from charging start time
df["Start Hour"] = df["Charging Start Time"].dt.hour


# ------------------------------------------------
# 4. SELECT FEATURES
# ------------------------------------------------

features = [
    "Vehicle Model",
    "Battery Capacity (kWh)",
    "Charging Station Location",
    "Charging Rate (kW)",
    "Time of Day",
    "Day of Week",
    "State of Charge (Start %)",
    "Distance Driven (since last charge) (km)",
    "Temperature (°C)",
    "Vehicle Age (years)",
    "Charger Type",
    "User Type",
    "Start Hour"
]

target = "Energy Consumed (kWh)"


# Remove rows where target value is missing
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# ------------------------------------------------
# 5. IDENTIFY COLUMN TYPES
# ------------------------------------------------

categorical_features = [
    "Vehicle Model",
    "Charging Station Location",
    "Time of Day",
    "Day of Week",
    "Charger Type",
    "User Type"
]

numerical_features = [
    "Battery Capacity (kWh)",
    "Charging Rate (kW)",
    "State of Charge (Start %)",
    "Distance Driven (since last charge) (km)",
    "Temperature (°C)",
    "Vehicle Age (years)",
    "Start Hour"
]


# ------------------------------------------------
# 6. PREPROCESSING
# ------------------------------------------------

# Numerical preprocessing
numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_transformer,
            numerical_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ------------------------------------------------
# 7. TRAIN / TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------
# 8. CREATE MODELS
# ------------------------------------------------

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=8
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=10
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ------------------------------------------------
# 9. TRAIN AND EVALUATE
# ------------------------------------------------

results = []


for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict
    predictions = pipeline.predict(
        X_test
    )

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# ------------------------------------------------
# 10. DISPLAY RESULTS
# ------------------------------------------------

results_df = pd.DataFrame(results)

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    results_df.to_string(
        index=False
    )
)


# ------------------------------------------------
# 11. FIND BEST MODEL
# ------------------------------------------------

best_model = results_df.loc[
    results_df["R2"].idxmax()
]

print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print(
    "Model:",
    best_model["Model"]
)

print(
    "MAE:",
    round(best_model["MAE"], 3)
)

print(
    "RMSE:",
    round(best_model["RMSE"], 3)
)

print(
    "R²:",
    round(best_model["R2"], 3)
)