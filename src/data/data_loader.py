"""
Data Loader for Air Quality Dataset
Downloads and caches air quality data from OpenAQ API and weather data from Open-Meteo API.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import pandas as pd
from pathlib import Path


class AirQualityDataLoader:
    """Handles downloading and caching of air quality and weather data."""
    
    def __init__(self, cache_dir: str = "data/raw"):
        """
        Initialize data loader with caching directory.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.openaq_base_url = "https://api.openaq.org/v2"
        self.meteo_base_url = "https://api.open-meteo.com/v1"
        
    def _get_cache_path(self, identifier: str) -> Path:
        """Generate cache file path for a given identifier."""
        return self.cache_dir / f"{identifier}.csv"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False
        
        age = time.time() - cache_path.stat().st_mtime
        return age < (max_age_hours * 3600)
    
    def get_air_quality_data(
        self, 
        city: str = "Los Angeles",
        country: str = "US",
        days: int = 90,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch air quality data for a specific city.
        
        Args:
            city: City name
            country: Country code (ISO 3166-1 alpha-2)
            days: Number of days of historical data
            force_refresh: Force download even if cache exists
            
        Returns:
            DataFrame with air quality measurements
        """
        cache_id = f"aqi_{city}_{country}_{days}d"
        cache_path = self._get_cache_path(cache_id)
        
        if not force_refresh and self._is_cache_valid(cache_path):
            print(f"Loading cached data for {city}, {country}")
            return pd.read_csv(cache_path, parse_dates=['datetime'])
        
        print(f"Downloading air quality data for {city}, {country}...")
        
        # Calculate date range
        date_to = datetime.now()
        date_from = date_to - timedelta(days=days)
        
        # OpenAQ API request
        params = {
            'city': city,
            'country': country,
            'date_from': date_from.strftime('%Y-%m-%d'),
            'date_to': date_to.strftime('%Y-%m-%d'),
            'limit': 10000,
            'parameter': ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        }
        
        try:
            response = requests.get(
                f"{self.openaq_base_url}/measurements",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            measurements = data.get('results', [])
            
            if not measurements:
                print(f"No data found for {city}, {country}. Using synthetic data...")
                return self._generate_synthetic_data(days)
            
            # Convert to DataFrame
            records = []
            for m in measurements:
                records.append({
                    'datetime': m['date']['utc'],
                    'parameter': m['parameter'],
                    'value': m['value'],
                    'unit': m['unit'],
                    'location': m.get('location', city)
                })
            
            df = pd.DataFrame(records)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Pivot to wide format
            df_pivot = df.pivot_table(
                index='datetime',
                columns='parameter',
                values='value',
                aggfunc='mean'
            ).reset_index()
            
            # Save to cache
            df_pivot.to_csv(cache_path, index=False)
            print(f"Cached data to {cache_path}")
            
            return df_pivot
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}. Using synthetic data...")
            return self._generate_synthetic_data(days)
    
    def get_weather_data(
        self,
        latitude: float = 34.05,
        longitude: float = -118.24,
        days: int = 90
    ) -> pd.DataFrame:
        """
        Fetch weather data from Open-Meteo API.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            days: Number of days of historical data
            
        Returns:
            DataFrame with weather data
        """
        print(f"Downloading weather data for ({latitude}, {longitude})...")
        
        date_to = datetime.now()
        date_from = date_to - timedelta(days=days)
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': date_from.strftime('%Y-%m-%d'),
            'end_date': date_to.strftime('%Y-%m-%d'),
            'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation',
            'timezone': 'auto'
        }
        
        try:
            response = requests.get(
                f"{self.meteo_base_url}/forecast",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            hourly = data.get('hourly', {})
            
            df = pd.DataFrame({
                'datetime': pd.to_datetime(hourly['time']),
                'temperature': hourly['temperature_2m'],
                'humidity': hourly['relative_humidity_2m'],
                'wind_speed': hourly['wind_speed_10m'],
                'precipitation': hourly['precipitation']
            })
            
            return df
            
        except Exception as e:
            print(f"Weather API failed: {e}. Using synthetic weather data...")
            return self._generate_synthetic_weather(days)
    
    def _generate_synthetic_data(self, days: int) -> pd.DataFrame:
        """Generate synthetic air quality data for demonstration."""
        import numpy as np
        
        dates = pd.date_range(
            end=datetime.now(),
            periods=days * 24,
            freq='H'
        )
        
        # Generate realistic AQI patterns with daily/weekly cycles
        np.random.seed(42)
        base_pm25 = 25 + 15 * np.sin(np.arange(len(dates)) * 2 * np.pi / 168)  # Weekly cycle
        noise = np.random.normal(0, 8, len(dates))
        
        df = pd.DataFrame({
            'datetime': dates,
            'pm25': np.clip(base_pm25 + noise, 0, 150),
            'pm10': np.clip((base_pm25 + noise) * 1.5, 0, 200),
            'o3': np.clip(30 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.normal(0, 5, len(dates)), 0, 100),
            'no2': np.clip(20 + np.random.normal(0, 5, len(dates)), 0, 80),
            'so2': np.clip(5 + np.random.normal(0, 2, len(dates)), 0, 30),
            'co': np.clip(0.4 + np.random.normal(0, 0.1, len(dates)), 0, 2)
        })
        
        return df
    
    def _generate_synthetic_weather(self, days: int) -> pd.DataFrame:
        """Generate synthetic weather data for demonstration."""
        import numpy as np
        
        dates = pd.date_range(
            end=datetime.now(),
            periods=days * 24,
            freq='H'
        )
        
        np.random.seed(42)
        
        df = pd.DataFrame({
            'datetime': dates,
            'temperature': 20 + 8 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.normal(0, 2, len(dates)),
            'humidity': np.clip(60 + np.random.normal(0, 15, len(dates)), 20, 100),
            'wind_speed': np.clip(np.abs(5 + np.random.normal(0, 3, len(dates))), 0, 30),
            'precipitation': np.clip(np.random.exponential(0.1, len(dates)), 0, 10)
        })
        
        return df
    
    def merge_datasets(
        self,
        aqi_df: pd.DataFrame,
        weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge air quality and weather datasets on datetime.
        
        Args:
            aqi_df: Air quality DataFrame
            weather_df: Weather DataFrame
            
        Returns:
            Merged DataFrame
        """
        # Round to nearest hour for merging
        aqi_df['datetime'] = pd.to_datetime(aqi_df['datetime']).dt.floor('H')
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).dt.floor('H')
        
        merged = pd.merge(aqi_df, weather_df, on='datetime', how='inner')
        merged = merged.sort_values('datetime').reset_index(drop=True)
        
        print(f"Merged dataset: {len(merged)} records")
        return merged


def main():
    """Example usage of the data loader."""
    loader = AirQualityDataLoader()
    
    # Download data for Los Angeles
    aqi_data = loader.get_air_quality_data(
        city="Los Angeles",
        country="US",
        days=90
    )
    
    weather_data = loader.get_weather_data(
        latitude=34.05,
        longitude=-118.24,
        days=90
    )
    
    # Merge datasets
    full_data = loader.merge_datasets(aqi_data, weather_data)
    
    # Save combined dataset
    output_path = "data/processed/combined_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_data.to_csv(output_path, index=False)
    
    print(f"\n✓ Data saved to {output_path}")
    print(f"  Shape: {full_data.shape}")
    print(f"  Date range: {full_data['datetime'].min()} to {full_data['datetime'].max()}")
    print(f"\nColumns: {', '.join(full_data.columns)}")


if __name__ == "__main__":
    main()
