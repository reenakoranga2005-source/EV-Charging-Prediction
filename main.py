from src.train_model import train_model
from src.evaluate_model import evaluate_model


print("EV CHARGING PREDICTION")

# Train model
model, X_test, y_test = train_model()

# Evaluate model
evaluate_model(
    model,
    X_test,
    y_test
)

print("\nProject completed successfully!")