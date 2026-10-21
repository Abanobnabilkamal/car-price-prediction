from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

# File paths
DATA_PATH = DATA_DIR / "car_price_prediction.csv"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
RESULTS_PATH = MODEL_DIR / "model_comparison.csv"
# Target column
TARGET = "Price"

# Columns to drop
DROP_COLS = ["ID", "Doors"]

# Numerical features
NUM_COLS = [
    "Prod. year",
    "Engine volume",
    "Mileage",
    "Cylinders",
    "Levy",
    "Airbags",
]

# Categorical features
CAT_COLS = [
    "Manufacturer",
    "Model",
    "Category",
    "Leather interior",
    "Fuel type",
    "Gear box type",
    "Drive wheels",
    "Wheel",
    "Color"
]

# Training configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5