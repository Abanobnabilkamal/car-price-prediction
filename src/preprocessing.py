import pandas as pd

from src.config import DATA_PATH, DROP_COLS, TARGET ,NUM_COLS,CAT_COLS


def load_and_clean_data(file_path=DATA_PATH):
    # Load dataset
    df = pd.read_csv(file_path)

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop unnecessary columns
    df = df.drop(columns=DROP_COLS)

    # Clean Levy
    df["Levy"] = df["Levy"].replace("-", 0)

    # Clean Mileage
    df["Mileage"] = df["Mileage"].str.replace("km", "", regex=False)

    # Clean Engine volume
    df["Engine volume"] = df["Engine volume"].str.replace(
        " Turbo", "", regex=False
    )

   
    df[NUM_COLS] = df[NUM_COLS].astype("float64")
    df[CAT_COLS] = df[CAT_COLS].astype("category")

    return df


def split_features_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y