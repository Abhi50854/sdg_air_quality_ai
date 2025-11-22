"""
Unit Tests for Data Pipeline
Tests data loading, preprocessing, and feature engineering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np
from data.data_loader import AirQualityDataLoader
from data.preprocessing import AirQualityPreprocessor


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_initialization(self):
        """Test loader initialization."""
        loader = AirQualityDataLoader(cache_dir="tests/cache")
        assert loader.cache_dir.exists()
    
    def test_cache_path_generation(self):
        """Test cache path generation."""
        loader = AirQualityDataLoader()
        path = loader._get_cache_path("test_data")
        assert path.name == "test_data.csv"
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        loader = AirQualityDataLoader()
        df = loader._generate_synthetic_data(days=7)
        
        assert isinstance(df, pd.DataFrame)
        assert 'datetime' in df.columns
        assert 'pm25' in df.columns
        assert len(df) == 7 * 24  # 7 days * 24 hours
        assert df['pm25'].min() >= 0
        assert df['pm25'].max() <= 200
    
    def test_air_quality_data_loading(self):
        """Test air quality data loading (uses synthetic data in tests)."""
        loader = AirQualityDataLoader()
        df = loader.get_air_quality_data(city="TestCity", country="US", days=3)
        
        assert not df.empty
        assert 'datetime' in df.columns
        assert 'pm25' in df.columns
    
    def test_weather_data_loading(self):
        """Test weather data loading."""
        loader = AirQualityDataLoader()
        df = loader.get_weather_data(latitude=34.05, longitude=-118.24, days=3)
        
        assert not df.empty
        assert 'datetime' in df.columns
        assert 'temperature' in df.columns
    
    def test_dataset_merging(self):
        """Test merging of AQI and weather datasets."""
        loader = AirQualityDataLoader()
        
        aqi_df = loader._generate_synthetic_data(days=3)
        weather_df = loader._generate_synthetic_weather(days=3)
        
        merged = loader.merge_datasets(aqi_df, weather_df)
        
        assert not merged.empty
        assert 'pm25' in merged.columns
        assert 'temperature' in merged.columns
        assert 'datetime' in merged.columns


class TestPreprocessor:
    """Test data preprocessing functionality."""
    
    def test_feature_engineering(self):
        """Test feature engineering creates expected columns."""
        loader = AirQualityDataLoader()
        df = loader._generate_synthetic_data(days=5)
        
        preprocessor = AirQualityPreprocessor()
        df_processed = preprocessor.engineer_features(df)
        
        # Check time features
        assert 'hour_sin' in df_processed.columns
        assert 'hour_cos' in df_processed.columns
        assert 'dow_sin' in df_processed.columns
        assert 'month_sin' in df_processed.columns
        
        # Check rolling features
        assert 'pm25_roll_mean_24h' in df_processed.columns
        assert 'pm25_roll_std_24h' in df_processed.columns
        
        # Check no NaN values
        assert df_processed.isnull().sum().sum() == 0
    
    def test_sequence_creation(self):
        """Test LSTM sequence creation."""
        loader = AirQualityDataLoader()
        aqi_df = loader._generate_synthetic_data(days=10)
        weather_df = loader._generate_synthetic_weather(days=10)
        merged = loader.merge_datasets(aqi_df, weather_df)
        
        preprocessor = AirQualityPreprocessor()
        processed = preprocessor.engineer_features(merged)
        
        X_train, y_train, X_test, y_test = preprocessor.prepare_sequences(
            processed,
            lookback=72,
            forecast_horizon=24,
            train_split=0.8
        )
        
        # Check shapes
        assert len(X_train.shape) == 3  # (samples, timesteps, features)
        assert X_train.shape[1] == 72   # lookback
        assert len(y_train.shape) == 2  # (samples, forecast_horizon)
        assert y_train.shape[1] == 24   # forecast_horizon
        
        # Check train/test split
        total_sequences = len(X_train) + len(X_test)
        assert len(X_train) == int(total_sequences * 0.8)
    
    def test_scaler_persistence(self, tmp_path):
        """Test saving and loading scalers."""
        preprocessor = AirQualityPreprocessor()
        
        # Create dummy data
        loader = AirQualityDataLoader()
        aqi_df = loader._generate_synthetic_data(days=5)
        weather_df = loader._generate_synthetic_weather(days=5)
        merged = loader.merge_datasets(aqi_df, weather_df)
        processed = preprocessor.engineer_features(merged)
        
        # Prepare sequences to fit scalers
        preprocessor.prepare_sequences(processed, lookback=24, forecast_horizon=12)
        
        # Save scalers
        save_dir = str(tmp_path / "scalers")
        preprocessor.save_scalers(output_dir=save_dir)
        
        # Load in new instance
        new_preprocessor = AirQualityPreprocessor()
        new_preprocessor.load_scalers(input_dir=save_dir)
        
        # Check loaded
        assert new_preprocessor.feature_scaler is not None
        assert new_preprocessor.target_scaler is not None
        assert new_preprocessor.feature_columns is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
