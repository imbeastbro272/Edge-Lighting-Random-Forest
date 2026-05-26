"""
LED Brightness Random Forest Controller
Uses Random Forest for adaptive brightness control with better generalization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle
import os
from datetime import datetime


def calculate_time_features(hour):
    """
    Convert hour to cyclic features and time period
    
    Args:
        hour (int): Hour of day (0-23)
        
    Returns:
        tuple: (sin_hour, cos_hour, time_period, period_name)
    """
    # Cyclic encoding for hour
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    
    # Time period encoding
    if 0 <= hour < 6:
        time_period = 0  # Early Morning
        period_name = "Early Morning"
    elif 6 <= hour < 12:
        time_period = 1  # Morning
        period_name = "Morning"
    elif 12 <= hour < 17:
        time_period = 2  # Afternoon
        period_name = "Afternoon"
    elif 17 <= hour < 21:
        time_period = 3  # Evening
        period_name = "Evening"
    else:
        time_period = 4  # Night
        period_name = "Night"
    
    return sin_hour, cos_hour, time_period, period_name


class LEDBrightnessRFController:
    """Random Forest-based LED Brightness Controller"""
    
    def __init__(self):
        self.model = None
        self.feature_names = ['ambient_light', 'motion_detected', 'sin_hour', 
                             'cos_hour', 'time_period', 'day_of_week']
        self.training_history = []
        self.online_learning_buffer = []
        self.buffer_max_size = 100
        
    def augment_data(self, df):
        """
        Augment training data for better coverage
        
        Args:
            df (pd.DataFrame): Original dataset
            
        Returns:
            pd.DataFrame: Augmented dataset
        """
        print("\n=== Data Augmentation ===")
        original_size = len(df)
        
        # Check for night + high ambient light scenario
        night_mask = df['time_period'] == 4
        ambient_75th = df['ambient_light'].quantile(0.75)
        high_ambient_night = df[night_mask & (df['ambient_light'] > ambient_75th)]
        
        if len(high_ambient_night) < 10:
            print(f"⚠️  Insufficient night + high ambient light data ({len(high_ambient_night)} samples)")
            print("   Generating synthetic data...")
            
            # Generate synthetic samples
            n_synthetic = 50
            synthetic_data = []
            
            for _ in range(n_synthetic):
                # Night hours (21-23, 0-5)
                hour = np.random.choice([21, 22, 23, 0, 1, 2, 3, 4, 5])
                sin_h, cos_h, tp, _ = calculate_time_features(hour)
                
                # High ambient light (street lights, moon, etc.)
                ambient = np.random.randint(600, 1000)
                
                # Motion detection (both cases)
                motion = np.random.choice([0, 1])
                
                # Day of week
                dow = np.random.randint(0, 7)
                
                # Target brightness: Lower when ambient is high at night
                # Motion: 30-50%, No motion: 10-30%
                if motion:
                    brightness = np.random.randint(30, 51)
                else:
                    brightness = np.random.randint(10, 31)
                
                synthetic_data.append({
                    'ambient_light': ambient,
                    'motion_detected': motion,
                    'sin_hour': sin_h,
                    'cos_hour': cos_h,
                    'time_period': tp,
                    'day_of_week': dow,
                    'led_brightness': brightness
                })
            
            synthetic_df = pd.DataFrame(synthetic_data)
            df = pd.concat([df, synthetic_df], ignore_index=True)
            print(f"   Added {n_synthetic} synthetic samples")
        
        # Add noise augmentation for robustness
        print("   Adding noise-augmented samples...")
        noise_samples = []
        sample_indices = np.random.choice(len(df), size=min(100, len(df)), replace=False)
        
        for idx in sample_indices:
            row = df.iloc[idx].copy()
            # Add small noise to continuous features
            row['ambient_light'] = max(0, row['ambient_light'] + np.random.normal(0, 20))
            row['sin_hour'] = np.clip(row['sin_hour'] + np.random.normal(0, 0.05), -1, 1)
            row['cos_hour'] = np.clip(row['cos_hour'] + np.random.normal(0, 0.05), -1, 1)
            row['led_brightness'] = np.clip(row['led_brightness'] + np.random.normal(0, 2), 0, 100)
            noise_samples.append(row)
        
        noise_df = pd.DataFrame(noise_samples)
        df = pd.concat([df, noise_df], ignore_index=True)
        
        print(f"✓ Dataset augmented: {original_size} → {len(df)} samples (+{len(df)-original_size})")
        
        return df
    
    def train(self, df, test_size=0.2, random_state=42, n_estimators=100, 
              max_depth=15, min_samples_split=5, min_samples_leaf=2):
        """
        Train Random Forest model
        
        Args:
            df (pd.DataFrame): Training dataset
            test_size (float): Test split ratio
            random_state (int): Random seed
            n_estimators (int): Number of trees in forest
            max_depth (int): Maximum tree depth
            min_samples_split (int): Minimum samples to split
            min_samples_leaf (int): Minimum samples in leaf
            
        Returns:
            dict: Training metrics
        """
        # Augment data
        df = self.augment_data(df)
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['led_brightness']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Initialize Random Forest
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )
        
        print(f"\n=== Training Random Forest ===")
        print(f"Parameters:")
        print(f"  - Trees: {n_estimators}")
        print(f"  - Max Depth: {max_depth}")
        print(f"  - Min Samples Split: {min_samples_split}")
        print(f"  - Min Samples Leaf: {min_samples_leaf}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Clip predictions to valid range
        y_train_pred = np.clip(y_train_pred, 0, 100)
        y_test_pred = np.clip(y_test_pred, 0, 100)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"\n=== Training Results ===")
        print(f"Train MAE: {train_mae:.2f}%")
        print(f"Test MAE: {test_mae:.2f}%")
        print(f"Train RMSE: {train_rmse:.2f}%")
        print(f"Test RMSE: {test_rmse:.2f}%")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        # Feature importance
        print(f"\n=== Feature Importance ===")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_samples': len(df)
        })
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance
        }
    
    def predict(self, ambient_light, motion_detected, sin_hour, cos_hour, 
                time_period, day_of_week):
        """
        Predict LED brightness
        
        Args:
            ambient_light (float): Ambient light level (lux)
            motion_detected (int): Motion detection (0 or 1)
            sin_hour (float): Sine of hour
            cos_hour (float): Cosine of hour
            time_period (int): Time period (0-4)
            day_of_week (int): Day of week (0-6)
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare input
        X = np.array([[ambient_light, motion_detected, sin_hour, cos_hour, 
                      time_period, day_of_week]])
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Get prediction from individual trees for uncertainty estimation
        tree_predictions = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
        prediction_std = np.std(tree_predictions)
        
        # Clip to valid range
        final_brightness = np.clip(prediction, 0, 100)
        
        return {
            'ml_predicted': round(prediction, 2),
            'final_brightness': int(round(final_brightness)),
            'uncertainty': round(prediction_std, 2),
            'confidence': round(100 - min(prediction_std * 2, 100), 2)
        }
    
    def add_online_sample(self, ambient_light, motion_detected, sin_hour, 
                         cos_hour, time_period, day_of_week, actual_brightness):
        """
        Add sample to online learning buffer
        
        Args:
            Features and actual brightness from user feedback
        """
        sample = {
            'ambient_light': ambient_light,
            'motion_detected': motion_detected,
            'sin_hour': sin_hour,
            'cos_hour': cos_hour,
            'time_period': time_period,
            'day_of_week': day_of_week,
            'led_brightness': actual_brightness
        }
        
        self.online_learning_buffer.append(sample)
        
        # Keep buffer size limited
        if len(self.online_learning_buffer) > self.buffer_max_size:
            self.online_learning_buffer.pop(0)
    
    def retrain_with_online_data(self):
        """
        Retrain model with online learning buffer
        
        Returns:
            dict: Retraining metrics
        """
        if len(self.online_learning_buffer) < 10:
            return {'status': 'insufficient_data', 'buffer_size': len(self.online_learning_buffer)}
        
        print(f"\n=== Online Retraining ===")
        print(f"Buffer size: {len(self.online_learning_buffer)} samples")
        
        # Convert buffer to DataFrame
        new_df = pd.DataFrame(self.online_learning_buffer)
        
        # Prepare features and target
        X_new = new_df[self.feature_names]
        y_new = new_df['led_brightness']
        
        # Incremental training (train on new data)
        self.model.fit(X_new, y_new)
        
        # Evaluate on new data
        y_pred = self.model.predict(X_new)
        y_pred = np.clip(y_pred, 0, 100)
        
        mae = mean_absolute_error(y_new, y_pred)
        r2 = r2_score(y_new, y_pred)
        
        print(f"Retrain MAE: {mae:.2f}%")
        print(f"Retrain R²: {r2:.4f}")
        
        return {
            'status': 'success',
            'mae': mae,
            'r2': r2,
            'n_samples': len(self.online_learning_buffer)
        }
    
    def save_model(self, filepath='led_rf_model.pkl'):
        """
        Save trained model to file
        
        Args:
            filepath (str): Output file path
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'online_buffer': self.online_learning_buffer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to: {filepath}")
        print(f"  File size: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    def load_model(self, filepath='led_rf_model.pkl'):
        """
        Load trained model from file
        
        Args:
            filepath (str): Model file path
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data.get('training_history', [])
        self.online_learning_buffer = model_data.get('online_buffer', [])
        
        print(f"✓ Model loaded from: {filepath}")
        print(f"  Trees in forest: {self.model.n_estimators}")
        print(f"  Training history: {len(self.training_history)} entries")
