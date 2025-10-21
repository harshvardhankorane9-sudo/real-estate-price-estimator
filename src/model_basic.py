import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.data_processor import clean_and_engineer_features, split_X_y
import math

def train_model(csv_path: str):
    """Train a linear regression model for house price prediction."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} rows from {csv_path}")

    # Clean and prepare data
    df_clean = clean_and_engineer_features(df)
    X, y = split_X_y(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define columns
    categorical_features = ['area_type', 'location']
    numeric_features = ['total_sqft', 'bath', 'bhk']

    # Column transformer and pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train the model
    model.fit(X_train, y_train)
    print("✅ Model training complete!")

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print(f"R²: {r2_score(y_test, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

    # Compute RMSE manually for backward compatibility
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.3f}")

    return model


if __name__ == "__main__":
    train_model("data/Bengaluru_House_Data.csv")
