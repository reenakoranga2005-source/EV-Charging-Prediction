import pandas as pd


DATA_PATH = "data/ev_workplace_charging_data.csv"


def load_data():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)

    # ------------------------------------------------
    # Create time-based features
    # ------------------------------------------------

    df["created"] = pd.to_datetime(df["created"])

    df["hour"] = df["created"].dt.hour

    df["day_of_week"] = df["created"].dt.dayofweek

    df["month"] = df["created"].dt.month

    # ------------------------------------------------
    # Define target
    # ------------------------------------------------

    target = "kwhTotal"

    # ------------------------------------------------
    # Select features
    # ------------------------------------------------

    features = [
        "stationId",
        "locationId",
        "platform",
        "userId",
        "managerVehicle",
        "facilityType",
        "reportedZip",
        "totalSessions",
        "habitualUser",
        "earlyAdopter",
        "hour",
        "day_of_week",
        "month"
    ]

    X = df[features]

    y = df[target]

    return X, y