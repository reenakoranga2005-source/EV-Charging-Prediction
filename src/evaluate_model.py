from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


def evaluate_model(model, X_test, y_test):

    # Prediction
    predictions = model.predict(X_test)

    # MAE
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    # R2
    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nModel Performance")
    print("------------------------")

    print("MAE  :", round(mae, 3))
    print("RMSE :", round(rmse, 3))
    print("R²   :", round(r2, 3))

    return mae, rmse, r2