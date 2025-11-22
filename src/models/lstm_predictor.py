"""
LSTM Model for Air Quality Index Prediction
Implements a deep LSTM neural network for time-series forecasting.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
from typing import Tuple, Optional


class LSTMAQIPredictor:
    """LSTM neural network for AQI prediction."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        forecast_horizon: int = 24,
        lstm_units: Tuple[int, int] = (128, 64),
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_shape: (lookback_steps, num_features)
            forecast_horizon: Number of future steps to predict
            lstm_units: Number of units in each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
        """
        self.input_shape = input_shape
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def build_model(self) -> keras.Model:
        """
        Build LSTM architecture.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(
                self.lstm_units[0],
                return_sequences=True,
                input_shape=self.input_shape,
                name='lstm_1'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_1'),
            
            # Second LSTM layer
            layers.LSTM(
                self.lstm_units[1],
                return_sequences=False,
                name='lstm_2'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_2'),
            
            # Dense layers for output
            layers.Dense(64, activation='relu', name='dense_1'),
            layers.Dropout(self.dropout_rate / 2, name='dropout_3'),
            
            # Output layer (forecast multiple steps)
            layers.Dense(self.forecast_horizon, name='output')
        ])
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae', self._rmse_metric]
        )
        
        self.model = model
        
        print("\n✓ Model Architecture:")
        model.summary()
        
        return model
    
    @staticmethod
    def _rmse_metric(y_true, y_pred):
        """RMSE metric for Keras."""
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    
    def get_callbacks(
        self,
        model_path: str = "models/saved/best_model.keras",
        patience: int = 15
    ) -> list:
        """
        Create training callbacks.
        
        Args:
            model_path: Path to save best model
            patience: Epochs to wait before early stopping
            
        Returns:
            List of Keras callbacks
        """
        return [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir='logs/tensorboard',
                histogram_freq=1
            )
        ]
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32
    ) -> keras.callbacks.History:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        print(f"\n🚀 Starting training...")
        print(f"  Epochs: {epochs}, Batch size: {batch_size}")
        print(f"  Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        print("\n✓ Training complete!")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() or load_model() first.")
        
        return self.model.predict(X)
    
    def save_model(self, path: str = "models/saved/final_model.keras"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train or build a model first.")
        
        self.model.save(path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str = "models/saved/final_model.keras"):
        """Load a pre-trained model."""
        self.model = keras.models.load_model(
            path,
            custom_objects={'_rmse_metric': self._rmse_metric}
        )
        print(f"✓ Model loaded from {path}")


def main():
    """Example usage of LSTM predictor."""
    # Load prepared sequences
    data = np.load("data/processed/sequences.npz")
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Data loaded:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    
    # Initialize model
    predictor = LSTMAQIPredictor(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        forecast_horizon=y_train.shape[1] if len(y_train.shape) > 1 else 1,
        lstm_units=(128, 64),
        dropout_rate=0.3
    )
    
    # Build model
    predictor.build_model()
    
    # Train model
    history = predictor.train(
        X_train, y_train,
        X_test, y_test,  # Using test as validation for demo
        epochs=50,
        batch_size=32
    )
    
    # Save model
    predictor.save_model()
    
    print("\n✓ Training complete! Model saved.")


if __name__ == "__main__":
    main()
