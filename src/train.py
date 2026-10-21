import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import (
    DATA_PATH,
    MODEL_PATH,
    MODEL_DIR,
    RESULTS_PATH,
    NUM_COLS,
    CAT_COLS,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
)

from src.preprocessing import load_and_clean_data , split_features_target


def train_models():
    # Load and clean data
    df = load_and_clean_data(DATA_PATH)

    # Split features and target
    X,y = split_features_target(df)

    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    # Models and hyperparameters
    models = {
        "Ridge": (Ridge(),{"model__alpha": [0.01, 0.1, 1, 10, 100]},),
        "RandomForest": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [1, 10, 20],
                "model__min_samples_split": [2, 5],
            },
        ),
        "GradientBoosting": (GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
            },
        ),
        "XGBoost": (XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [100, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 6],
                "model__subsample": [0.8, 1.0],
            },
        ),
    }

    results = []
    best_models = {}

    # Train and evaluate models
    for name, (estimator, params) in models.items():
        print(f"Training {name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        grid = GridSearchCV(
            pipeline,
            param_grid=params,
            cv=CV_FOLDS,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        predictions = best_model.predict(X_test)

        best_models[name] = best_model

        results.append(
            {
                "Model": name,
                "Best Params": grid.best_params_,
                "CV RMSE": -grid.best_score_,
                "Test RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                "Test MAE": mean_absolute_error(y_test, predictions),
                "Test R2": r2_score(y_test, predictions),
            }
        )

    # Compare results
    results_df = pd.DataFrame(results).sort_values("CV RMSE")

    print("\nModel Comparison:")
    print(results_df.drop(columns="Best Params").to_string(index=False))

    # Save model comparison results
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\nComparison results saved to: {RESULTS_PATH}")

    # Select the best model
    best_name = results_df.iloc[0]["Model"]
    best_model = best_models[best_name]

    print(f"\nBest model: {best_name}")
    print("Best parameters:", results_df.iloc[0]["Best Params"])

    # Save the complete pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    train_models()