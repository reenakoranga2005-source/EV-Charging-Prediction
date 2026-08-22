import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from .data_preprocessing import load_data
from .data_preprocessing import clean_data
from .data_preprocessing import prepare_features


DATA_PATH = "C:/Users/mysel/OneDrive/Desktop/ev_charging prediction/data/ev_charging_patterns.csv"
MODEL_PATH = "models/energy_model.pkl"


def train_model():

    # Load dataset
    df = load_data(DATA_PATH)

    # Clean dataset
    df = clean_data(df)

    # Prepare features
    X, y = prepare_features(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Create model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)

    print("Model trained successfully")
    print("Model saved to:", MODEL_PATH)

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()