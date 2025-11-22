"""
Data Preprocessing for Air Quality Prediction
Handles feature engineering, normalization, and sequence creation for LSTM models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import joblib
from pathlib import Path


class AirQualityPreprocessor:
    """Preprocesses air quality data for LSTM model training."""
    
    def __init__(self):
        """Initialize preprocessor with scalers."""
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'pm25'
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based and statistical features.
        
        Args:
            df: Raw DataFrame with datetime and measurements
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Time-based features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['month'] = df['datetime'].dt.month
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Rolling statistics (if pm25 exists)
        if 'pm25' in df.columns:
            df['pm25_roll_mean_24h'] = df['pm25'].rolling(window=24, min_periods=1).mean()
            df['pm25_roll_std_24h'] = df['pm25'].rolling(window=24, min_periods=1).std().fillna(0)
            df['pm25_roll_max_24h'] = df['pm25'].rolling(window=24, min_periods=1).max()
            df['pm25_roll_min_24h'] = df['pm25'].rolling(window=24, min_periods=1).min()
        
        # Weather interaction features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
        
        if 'wind_speed' in df.columns:
            df['wind_speed_squared'] = df['wind_speed'] ** 2
        
        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        lookback: int = 72,
        forecast_horizon: int = 24,
        train_split: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            df: DataFrame with engineered features
            lookback: Number of past hours to use as input
            forecast_horizon: Number of future hours to predict
            train_split: Fraction of data for training
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Select feature columns (exclude datetime and target)
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime', self.target_column]]
        
        self.feature_columns = feature_cols
        
        # Extract features and target
        X = df[feature_cols].values
        y = df[self.target_column].values.reshape(-1, 1)
        
        # Fit scalers on training data only
        split_idx = int(len(X) * train_split)
        
        self.feature_scaler.fit(X[:split_idx])
        self.target_scaler.fit(y[:split_idx])
        
        # Transform all data
        X_scaled = self.feature_scaler.transform(X)
        y_scaled = self.target_scaler.transform(y)
        
        # Create sequences
        X_seq, y_seq = [], []
        
        for i in range(lookback, len(X_scaled) - forecast_horizon + 1):
            X_seq.append(X_scaled[i - lookback:i])
            y_seq.append(y_scaled[i:i + forecast_horizon])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq).squeeze()
        
        # Split into train/test
        split_idx = int(len(X_seq) * train_split)
        
        X_train = X_seq[:split_idx]
        y_train = y_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_test = y_seq[split_idx:]
        
        print(f"✓ Created sequences:")
        print(f"  Lookback: {lookback} hours, Forecast: {forecast_horizon} hours")
        print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def save_scalers(self, output_dir: str = "models/saved"):
        """Save fitted scalers for later use."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.feature_scaler, f"{output_dir}/feature_scaler.pkl")
        joblib.dump(self.target_scaler, f"{output_dir}/target_scaler.pkl")
        joblib.dump(self.feature_columns, f"{output_dir}/feature_columns.pkl")
        
        print(f"✓ Scalers saved to {output_dir}")
    
    def load_scalers(self, input_dir: str = "models/saved"):
        """Load previously saved scalers."""
        self.feature_scaler = joblib.load(f"{input_dir}/feature_scaler.pkl")
        self.target_scaler = joblib.load(f"{input_dir}/target_scaler.pkl")
        self.feature_columns = joblib.load(f"{input_dir}/feature_columns.pkl")
        
        print(f"✓ Scalers loaded from {input_dir}")
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original scale."""
        return self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()


def main():
    """Example usage of the preprocessor."""
    # Load data
    df = pd.read_csv("data/processed/combined_data.csv")
    
    # Initialize preprocessor
    preprocessor = AirQualityPreprocessor()
    
    # Engineer features
    df_processed = preprocessor.engineer_features(df)
    
    print("\n✓ Feature engineering complete")
    print(f"  Original columns: {df.shape[1]}")
    print(f"  Processed columns: {df_processed.shape[1]}")
    
    # Create sequences
    X_train, y_train, X_test, y_test = preprocessor.prepare_sequences(
        df_processed,
        lookback=72,
        forecast_horizon=24
    )
    
    # Save scalers
    preprocessor.save_scalers()
    
    # Save processed data
    np.savez(
        "data/processed/sequences.npz",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    print("\n✓ Sequences saved to data/processed/sequences.npz")


if __name__ == "__main__":
    main()
