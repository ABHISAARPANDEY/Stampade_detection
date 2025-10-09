"""
Predictive Analytics Module for STAMPede Detection System
Implements machine learning models for crowd behavior prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from collections import deque
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

@dataclass
class PredictionResult:
    timestamp: float
    camera_id: int
    prediction_type: str
    predicted_value: float
    confidence: float
    time_horizon: int  # seconds into the future
    features_used: List[str]
    model_name: str

@dataclass
class TrendAnalysis:
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1
    change_rate: float
    predicted_peak: Optional[float]
    predicted_peak_time: Optional[float]
    confidence: float

class CrowdPredictor:
    """Predicts crowd behavior using machine learning models"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Data storage
        self.historical_data = deque(maxlen=10000)  # Store last 10k records
        self.feature_scalers = {}
        self.models = {}
        
        # Model configurations
        self.model_configs = {
            'density_prediction': {
                'model_class': RandomForestRegressor,
                'params': {'n_estimators': 100, 'random_state': 42},
                'features': ['people_count', 'density', 'flow_intensity', 'hour', 'day_of_week']
            },
            'people_count_prediction': {
                'model_class': GradientBoostingRegressor,
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
                'features': ['density', 'flow_intensity', 'movement_risk_score', 'hour', 'day_of_week']
            },
            'risk_prediction': {
                'model_class': Ridge,
                'params': {'alpha': 1.0},
                'features': ['people_count', 'density', 'flow_intensity', 'movement_risk_score', 'hour']
            }
        }
        
        # Load existing models
        self._load_models()
    
    def add_data_point(self, camera_id: int, people_count: int, density: float,
                      flow_intensity: float, movement_risk_score: float,
                      timestamp: Optional[float] = None):
        """Add a new data point for training"""
        if timestamp is None:
            timestamp = time.time()
        
        # Extract time features
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        day_of_week = dt.weekday()
        
        data_point = {
            'timestamp': timestamp,
            'camera_id': camera_id,
            'people_count': people_count,
            'density': density,
            'flow_intensity': flow_intensity,
            'movement_risk_score': movement_risk_score,
            'hour': hour,
            'day_of_week': day_of_week
        }
        
        self.historical_data.append(data_point)
    
    def prepare_features(self, data: List[Dict], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        if not data:
            return np.array([]), np.array([])
        
        df = pd.DataFrame(data)
        
        # Get feature columns
        feature_cols = ['people_count', 'density', 'flow_intensity', 'movement_risk_score', 'hour', 'day_of_week']
        
        # Prepare features
        X = df[feature_cols].values
        y = df[target_col].values
        
        return X, y
    
    def train_model(self, model_name: str, target_col: str) -> bool:
        """Train a specific model"""
        if model_name not in self.model_configs:
            print(f"[PredictiveAnalytics] Unknown model: {model_name}")
            return False
        
        # Prepare data
        data = list(self.historical_data)
        if len(data) < 100:  # Need minimum data for training
            print(f"[PredictiveAnalytics] Insufficient data for training {model_name}")
            return False
        
        X, y = self.prepare_features(data, target_col)
        
        if len(X) == 0:
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        config = self.model_configs[model_name]
        model = config['model_class'](**config['params'])
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"[PredictiveAnalytics] {model_name} trained - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save model and scaler
        self.models[model_name] = model
        self.feature_scalers[model_name] = scaler
        
        # Save to disk
        self._save_model(model_name, model, scaler)
        
        return True
    
    def predict_density(self, camera_id: int, time_horizon: int = 60) -> Optional[PredictionResult]:
        """Predict density for a specific camera"""
        return self._predict('density_prediction', 'density', camera_id, time_horizon)
    
    def predict_people_count(self, camera_id: int, time_horizon: int = 60) -> Optional[PredictionResult]:
        """Predict people count for a specific camera"""
        return self._predict('people_count_prediction', 'people_count', camera_id, time_horizon)
    
    def predict_risk(self, camera_id: int, time_horizon: int = 60) -> Optional[PredictionResult]:
        """Predict risk score for a specific camera"""
        return self._predict('risk_prediction', 'movement_risk_score', camera_id, time_horizon)
    
    def _predict(self, model_name: str, target_col: str, camera_id: int, time_horizon: int) -> Optional[PredictionResult]:
        """Make a prediction using a specific model"""
        if model_name not in self.models:
            print(f"[PredictiveAnalytics] Model {model_name} not trained")
            return None
        
        # Get recent data for this camera
        recent_data = [
            d for d in self.historical_data 
            if d['camera_id'] == camera_id
        ][-10:]  # Last 10 data points
        
        if len(recent_data) < 3:
            print(f"[PredictiveAnalytics] Insufficient data for prediction")
            return None
        
        # Prepare features
        X, _ = self.prepare_features(recent_data, target_col)
        if len(X) == 0:
            return None
        
        # Use most recent data point for prediction
        latest_features = X[-1:].copy()
        
        # Adjust time features for future prediction
        future_time = time.time() + time_horizon
        future_dt = datetime.fromtimestamp(future_time)
        latest_features[0][4] = future_dt.hour  # hour
        latest_features[0][5] = future_dt.weekday()  # day_of_week
        
        # Scale features
        scaler = self.feature_scalers[model_name]
        features_scaled = scaler.transform(latest_features)
        
        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(features_scaled)[0]
        
        # Calculate confidence (simplified - could be improved with uncertainty quantification)
        confidence = min(0.9, max(0.1, len(recent_data) / 10.0))
        
        return PredictionResult(
            timestamp=time.time(),
            camera_id=camera_id,
            prediction_type=target_col,
            predicted_value=float(prediction),
            confidence=confidence,
            time_horizon=time_horizon,
            features_used=self.model_configs[model_name]['features'],
            model_name=model_name
        )
    
    def analyze_trends(self, camera_id: int, window_minutes: int = 30) -> Optional[TrendAnalysis]:
        """Analyze trends in crowd behavior"""
        # Get recent data
        cutoff_time = time.time() - (window_minutes * 60)
        recent_data = [
            d for d in self.historical_data 
            if d['camera_id'] == camera_id and d['timestamp'] >= cutoff_time
        ]
        
        if len(recent_data) < 5:
            return None
        
        # Extract density values
        densities = [d['density'] for d in recent_data]
        timestamps = [d['timestamp'] for d in recent_data]
        
        # Calculate trend
        if len(densities) >= 3:
            # Simple linear trend
            x = np.array(timestamps)
            y = np.array(densities)
            
            # Normalize timestamps
            x_norm = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else x
            
            # Linear regression
            coeffs = np.polyfit(x_norm, y, 1)
            slope = coeffs[0]
            
            # Determine trend direction
            if slope > 0.1:
                trend_direction = 'increasing'
            elif slope < -0.1:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
            
            # Calculate trend strength
            trend_strength = min(1.0, abs(slope) * 10)
            
            # Calculate change rate
            change_rate = (densities[-1] - densities[0]) / max(densities[0], 0.1)
            
            # Predict peak (simplified)
            predicted_peak = None
            predicted_peak_time = None
            if trend_direction == 'increasing':
                # Simple linear extrapolation
                predicted_peak = densities[-1] + slope * 2  # 2 time units ahead
                predicted_peak_time = timestamps[-1] + 2 * (timestamps[-1] - timestamps[0]) / len(timestamps)
            
            # Calculate confidence
            r_squared = self._calculate_r_squared(x_norm, y, coeffs)
            confidence = max(0.1, min(0.9, r_squared))
            
            return TrendAnalysis(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                change_rate=change_rate,
                predicted_peak=predicted_peak,
                predicted_peak_time=predicted_peak_time,
                confidence=confidence
            )
        
        return None
    
    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> float:
        """Calculate R-squared for linear regression"""
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def get_predictions_summary(self, camera_id: int) -> Dict[str, Any]:
        """Get summary of all predictions for a camera"""
        predictions = {
            'density': self.predict_density(camera_id),
            'people_count': self.predict_people_count(camera_id),
            'risk': self.predict_risk(camera_id)
        }
        
        trends = self.analyze_trends(camera_id)
        
        return {
            'predictions': {k: v.__dict__ if v else None for k, v in predictions.items()},
            'trends': trends.__dict__ if trends else None,
            'timestamp': time.time()
        }
    
    def _save_model(self, model_name: str, model: Any, scaler: Any):
        """Save model and scaler to disk"""
        model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
        
        try:
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            print(f"[PredictiveAnalytics] Saved {model_name} model")
        except Exception as e:
            print(f"[PredictiveAnalytics] Failed to save {model_name}: {e}")
    
    def _load_models(self):
        """Load existing models from disk"""
        for model_name in self.model_configs.keys():
            model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
            
            try:
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    
                    self.models[model_name] = model
                    self.feature_scalers[model_name] = scaler
                    print(f"[PredictiveAnalytics] Loaded {model_name} model")
            except Exception as e:
                print(f"[PredictiveAnalytics] Failed to load {model_name}: {e}")
    
    def retrain_models(self):
        """Retrain all models with current data"""
        print("[PredictiveAnalytics] Retraining all models...")
        
        success_count = 0
        for model_name, config in self.model_configs.items():
            if model_name == 'density_prediction':
                target_col = 'density'
            elif model_name == 'people_count_prediction':
                target_col = 'people_count'
            elif model_name == 'risk_prediction':
                target_col = 'movement_risk_score'
            else:
                continue
            
            if self.train_model(model_name, target_col):
                success_count += 1
        
        print(f"[PredictiveAnalytics] Retrained {success_count}/{len(self.model_configs)} models")
        return success_count
    
    def get_model_performance(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get performance metrics for a model"""
        if model_name not in self.models:
            return None
        
        # This would require validation data - simplified for now
        return {
            'status': 'trained',
            'data_points': len(self.historical_data),
            'last_trained': time.time()
        }
    
    def export_data(self, filepath: str) -> bool:
        """Export historical data to CSV"""
        try:
            df = pd.DataFrame(list(self.historical_data))
            df.to_csv(filepath, index=False)
            print(f"[PredictiveAnalytics] Exported data to {filepath}")
            return True
        except Exception as e:
            print(f"[PredictiveAnalytics] Failed to export data: {e}")
            return False
    
    def import_data(self, filepath: str) -> bool:
        """Import historical data from CSV"""
        try:
            df = pd.read_csv(filepath)
            data = df.to_dict('records')
            
            # Clear existing data
            self.historical_data.clear()
            
            # Add imported data
            for record in data:
                self.historical_data.append(record)
            
            print(f"[PredictiveAnalytics] Imported {len(data)} records from {filepath}")
            return True
        except Exception as e:
            print(f"[PredictiveAnalytics] Failed to import data: {e}")
            return False
