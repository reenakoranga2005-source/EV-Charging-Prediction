import pandas as pd

def load_data(file_path="C:/Users/mysel/OneDrive/Desktop/ev_charging prediction/data/ev_charging_patterns.csv"):
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    return df

def clean_data(df):
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Remove rows where target is missing
    df = df.dropna(subset=['Energy Consumed (kWh)', 'Charging Duration (hours)'])
    
    # Fill missing numerical values
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())
        
    # Fill missing categorical values
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])
        
    return df

def prepare_features(df):

    # Create SOC difference
    df["SOC Difference"] = (
        df["State of Charge (End %)"]
        - df["State of Charge (Start %)"]
    )

    # Create estimated energy requirement
    df["Estimated Energy (kWh)"] = (
        df["Battery Capacity (kWh)"]
        * df["SOC Difference"]
        / 100
    )

    features = [
        "Battery Capacity (kWh)",
        "Charging Rate (kW)",
        "State of Charge (Start %)",
        "State of Charge (End %)",
        "SOC Difference",
        "Estimated Energy (kWh)",
        "Distance Driven (since last charge) (km)",
        "Temperature (°C)",
        "Vehicle Age (years)"
    ]

    X = df[features]

    y = df["Energy Consumed (kWh)"]

    return X, y
