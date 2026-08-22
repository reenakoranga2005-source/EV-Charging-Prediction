import joblib
import pandas as pd


MODEL_PATH = "models/energy_model.pkl"


def predict_energy(
    battery_capacity,
    charging_rate,
    start_soc,
    end_soc,
    distance_driven,
    temperature,
    vehicle_age
):

    # Calculate SOC difference
    soc_difference = end_soc - start_soc

    # Calculate estimated energy
    estimated_energy = (
        battery_capacity
        * soc_difference
        / 100
    )

    # Create input data
    new_data = pd.DataFrame({
        "Battery Capacity (kWh)": [
            battery_capacity
        ],

        "Charging Rate (kW)": [
            charging_rate
        ],

        "State of Charge (Start %)": [
            start_soc
        ],

        "State of Charge (End %)": [
            end_soc
        ],

        "SOC Difference": [
            soc_difference
        ],

        "Estimated Energy (kWh)": [
            estimated_energy
        ],

        "Distance Driven (since last charge) (km)": [
            distance_driven
        ],

        "Temperature (°C)": [
            temperature
        ],

        "Vehicle Age (years)": [
            vehicle_age
        ]
    })

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Make prediction
    prediction = model.predict(new_data)

    return prediction[0]


if __name__ == "__main__":

    energy = predict_energy(
        battery_capacity=60,
        charging_rate=20,
        start_soc=30,
        end_soc=80,
        distance_driven=100,
        temperature=25,
        vehicle_age=3
    )

    print()
    print("EV CHARGING PREDICTION")
    print("------------------------------")
    print(
        "Predicted Energy Consumption:",
        round(energy, 2),
        "kWh"
    )