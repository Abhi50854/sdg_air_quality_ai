"""
Model Evaluation and Bias Testing
Evaluates LSTM model performance and checks for bias across different demographics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data.preprocessing import AirQualityPreprocessor
from models.lstm_predictor import LSTMAQIPredictor
import matplotlib.pyplot as plt
import json


class ModelEvaluator:
    """Evaluates model performance and fairness."""
    
    def __init__(self, model_path: str = "models/saved/final_model.keras"):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
        """
        self.model = LSTMAQIPredictor(input_shape=(72, 20), forecast_horizon=24)
        self.model.load_model(model_path)
        
        self.preprocessor = AirQualityPreprocessor()
        self.preprocessor.load_scalers()
        
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label: str = "Overall"
    ) -> dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            label: Label for this evaluation
            
        Returns:
            Dictionary of metrics
        """
        # Flatten if multi-step predictions
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        metrics = {
            'label': label,
            'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
            'mae': mean_absolute_error(y_true_flat, y_pred_flat),
            'r2': r2_score(y_true_flat, y_pred_flat),
            'mape': np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
        }
        
        return metrics
    
    def evaluate_test_set(self) -> dict:
        """Evaluate on test set."""
        # Load test data
        data = np.load("data/processed/sequences.npz")
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test)
        
        # Inverse transform
        y_pred = self.preprocessor.inverse_transform_predictions(y_pred_scaled)
        y_true = self.preprocessor.inverse_transform_predictions(y_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, label="Test Set")
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"\n📊 {metrics['label']}:")
        print(f"  RMSE: {metrics['rmse']:.2f} AQI units")
        print(f"  MAE:  {metrics['mae']:.2f} AQI units")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        
        # Generate visualizations
        self._plot_predictions(y_true, y_pred)
        self._plot_residuals(y_true, y_pred)
        
        return metrics
    
    def bias_test(self) -> dict:
        """
        Test for bias across different time periods and conditions.
        
        Returns:
            Dictionary of bias metrics
        """
        print("\n" + "="*60)
        print("BIAS TESTING")
        print("="*60)
        
        # Load full data
        df = pd.read_csv("data/processed/combined_data.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Load test sequences
        data = np.load("data/processed/sequences.npz")
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Get predictions
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.preprocessor.inverse_transform_predictions(y_pred_scaled)
        y_true = self.preprocessor.inverse_transform_predictions(y_test)
        
        # Split test set by conditions
        test_size = len(y_test)
        test_datetimes = df['datetime'].iloc[-test_size:].values
        
        # Group 1: Weekday vs Weekend
        df_test = pd.DataFrame({'datetime': test_datetimes})
        df_test['datetime'] = pd.to_datetime(df_test['datetime'])
        df_test['is_weekend'] = df_test['datetime'].dt.dayofweek >= 5
        
        bias_results = {}
        
        # Weekend vs Weekday
        weekend_idx = df_test['is_weekend'].values[:len(y_true)]
        weekday_idx = ~weekend_idx
        
        if np.sum(weekend_idx) > 0:
            weekend_metrics = self.calculate_metrics(
                y_true[weekend_idx],
                y_pred[weekend_idx],
                label="Weekend"
            )
            bias_results['weekend'] = weekend_metrics
        
        if np.sum(weekday_idx) > 0:
            weekday_metrics = self.calculate_metrics(
                y_true[weekday_idx],
                y_pred[weekday_idx],
                label="Weekday"
            )
            bias_results['weekday'] = weekday_metrics
        
        # Print bias analysis
        print("\n🔍 Bias Analysis:")
        if 'weekend' in bias_results and 'weekday' in bias_results:
            rmse_diff = abs(bias_results['weekend']['rmse'] - bias_results['weekday']['rmse'])
            rmse_variance = (rmse_diff / bias_results['weekday']['rmse']) * 100
            
            print(f"\n  Weekend RMSE: {bias_results['weekend']['rmse']:.2f}")
            print(f"  Weekday RMSE: {bias_results['weekday']['rmse']:.2f}")
            print(f"  Variance: {rmse_variance:.1f}%")
            
            if rmse_variance < 10:
                print("  ✓ Bias test PASSED (variance < 10%)")
            else:
                print("  ⚠ Bias test WARNING (variance >= 10%)")
        
        # Save results
        with open("reports/evaluation_metrics.json", "w") as f:
            json.dump({
                'test_set': self.calculate_metrics(y_true, y_pred).__dict__,
                'bias_analysis': {k: v for k, v in bias_results.items()}
            }, f, indent=2)
        
        return bias_results
    
    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(12, 6))
        
        # Take first 200 predictions for visibility
        n_samples = min(200, len(y_true))
        x = np.arange(n_samples)
        
        plt.plot(x, y_true[:n_samples], label='Actual', alpha=0.7, linewidth=2)
        plt.plot(x, y_pred[:n_samples], label='Predicted', alpha=0.7, linewidth=2)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('PM2.5 AQI', fontsize=12)
        plt.title('Actual vs Predicted Air Quality Index', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('reports/predictions_plot.png', dpi=150)
        print("\n✓ Saved predictions plot to reports/predictions_plot.png")
        plt.close()
    
    def _plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot residuals distribution."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals scatter
        axes[0].scatter(y_pred, residuals, alpha=0.4, s=10)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title('Residual Plot', fontsize=13, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residual Value', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/residuals_plot.png', dpi=150)
        print("✓ Saved residuals plot to reports/residuals_plot.png")
        plt.close()


def main():
    """Run model evaluation."""
    import os
    os.makedirs("reports", exist_ok=True)
    
    evaluator = ModelEvaluator()
    
    # Evaluate test set
    test_metrics = evaluator.evaluate_test_set()
    
    # Bias testing
    bias_results = evaluator.bias_test()
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE!")
    print("="*60)
    print("\nOutputs:")
    print("  📊 Metrics: reports/evaluation_metrics.json")
    print("  📈 Predictions: reports/predictions_plot.png")
    print("  📉 Residuals: reports/residuals_plot.png")


if __name__ == "__main__":
    main()
