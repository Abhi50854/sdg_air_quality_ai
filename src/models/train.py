"""
Training Pipeline for AQI Prediction Model
Orchestrates data loading, preprocessing, model training, and experiment tracking.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import AirQualityDataLoader
from data.preprocessing import AirQualityPreprocessor
from models.lstm_predictor import LSTMAQIPredictor
import numpy as np


class TrainingPipeline:
    """End-to-end training pipeline."""
    
    def __init__(self, config: dict = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.loader = AirQualityDataLoader()
        self.preprocessor = AirQualityPreprocessor()
        self.model = None
        
    @staticmethod
    def _default_config() -> dict:
        """Default configuration."""
        return {
            'city': 'Los Angeles',
            'country': 'US',
            'latitude': 34.05,
            'longitude': -118.24,
            'days': 90,
            'lookback': 72,
            'forecast_horizon': 24,
            'train_split': 0.8,
            'lstm_units': (128, 64),
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32
        }
    
    def load_data(self):
        """Load and merge air quality and weather data."""
        print("\n" + "="*60)
        print("STEP 1: LOADING DATA")
        print("="*60)
        
        # Load air quality data
        aqi_data = self.loader.get_air_quality_data(
            city=self.config['city'],
            country=self.config['country'],
            days=self.config['days']
        )
        
        # Load weather data
        weather_data = self.loader.get_weather_data(
            latitude=self.config['latitude'],
            longitude=self.config['longitude'],
            days=self.config['days']
        )
        
        # Merge datasets
        self.data = self.loader.merge_datasets(aqi_data, weather_data)
        
        # Save combined data
        os.makedirs("data/processed", exist_ok=True)
        self.data.to_csv("data/processed/combined_data.csv", index=False)
        
        return self.data
    
    def preprocess_data(self):
        """Engineer features and create sequences."""
        print("\n" + "="*60)
        print("STEP 2: PREPROCESSING DATA")
        print("="*60)
        
        # Engineer features
        df_processed = self.preprocessor.engineer_features(self.data)
        
        print(f"\n✓ Feature engineering complete")
        print(f"  Original features: {self.data.shape[1]}")
        print(f"  Engineered features: {df_processed.shape[1]}")
        
        # Create sequences
        X_train, y_train, X_test, y_test = self.preprocessor.prepare_sequences(
            df_processed,
            lookback=self.config['lookback'],
            forecast_horizon=self.config['forecast_horizon'],
            train_split=self.config['train_split']
        )
        
        # Save sequences and scalers
        os.makedirs("data/processed", exist_ok=True)
        np.savez(
            "data/processed/sequences.npz",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        self.preprocessor.save_scalers()
        
        return X_train, y_train, X_test, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Build and train LSTM model."""
        print("\n" + "="*60)
        print("STEP 3: TRAINING MODEL")
        print("="*60)
        
        # Initialize model
        self.model = LSTMAQIPredictor(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            forecast_horizon=self.config['forecast_horizon'],
            lstm_units=self.config['lstm_units'],
            dropout_rate=self.config['dropout_rate'],
            learning_rate=self.config['learning_rate']
        )
        
        # Build architecture
        self.model.build_model()
        
        # Train
        history = self.model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size']
        )
        
        # Save model
        os.makedirs("models/saved", exist_ok=True)
        self.model.save_model("models/saved/final_model.keras")
        
        return history
    
    def run(self):
        """Execute full training pipeline."""
        print("\n" + "="*70)
        print(" 🌍 AIR QUALITY PREDICTION - TRAINING PIPELINE ")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Preprocess
        X_train, y_train, X_test, y_test = self.preprocess_data()
        
        # Train
        history = self.train_model(X_train, y_train, X_test, y_test)
        
        print("\n" + "="*70)
        print(" ✓ TRAINING PIPELINE COMPLETE!")
        print("="*70)
        print("\nOutputs:")
        print("  📊 Data: data/processed/combined_data.csv")
        print("  📦 Sequences: data/processed/sequences.npz")
        print("  🤖 Model: models/saved/final_model.keras")
        print("  📈 Scalers: models/saved/*.pkl")
        print("  📉 Logs: logs/tensorboard/")
        
        return history


def main():
    """Run training pipeline with default configuration."""
    pipeline = TrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
