"""
Training script for LED Brightness Control using Random Forest
Loads dataset, trains model, exports for ESP32 deployment
"""

import pandas as pd
import numpy as np
from led_brightness_rf_model import LEDBrightnessRFController, calculate_time_features
import sys
import json


def load_and_validate_data(filepath):
    """
    Load and validate the dataset
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Validated dataset
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Expected columns
    required_columns = ['ambient_light', 'motion_detected', 'sin_hour', 'cos_hour', 'time_period', 'led_brightness']
    optional_columns = ['day_of_week']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"\n⚠️  Warning: Missing columns: {missing_columns}")
        print("Please ensure your dataset has all required columns.")
        sys.exit(1)
    
    # Add day_of_week if not present
    if 'day_of_week' not in df.columns:
        print("\n⚠️  'day_of_week' column not found in dataset.")
        print("   Adding random day_of_week values for training...")
        df['day_of_week'] = np.random.randint(0, 7, size=len(df))
    
    required_columns.append('day_of_week')
    
    # Data validation
    print("\n=== Data Statistics ===")
    print(df[required_columns].describe())
    
    # Check for missing values
    missing_values = df[required_columns].isnull().sum()
    if missing_values.any():
        print("\n⚠️  Missing values detected:")
        print(missing_values[missing_values > 0])
        print("Dropping rows with missing values...")
        df = df.dropna(subset=required_columns)
    
    # Validate ranges
    print("\n=== Data Validation ===")
    
    # Check LED brightness range
    if df['led_brightness'].min() < 0 or df['led_brightness'].max() > 100:
        print(f"⚠️  LED brightness out of range [0, 100]: [{df['led_brightness'].min()}, {df['led_brightness'].max()}]")
    else:
        print(f"✓ LED brightness range: [{df['led_brightness'].min():.2f}, {df['led_brightness'].max():.2f}]")
    
    # Check motion_detected is binary
    unique_motion = df['motion_detected'].unique()
    if not all(val in [0, 1] for val in unique_motion):
        print(f"⚠️  motion_detected should be binary (0 or 1), found: {unique_motion}")
    else:
        print(f"✓ motion_detected is binary: {unique_motion}")
    
    # Check time_period range
    unique_periods = sorted(df['time_period'].unique())
    print(f"✓ Time periods found: {unique_periods}")
    
    # Distribution analysis
    print("\n=== Data Distribution ===")
    print("\nTime Period Distribution:")
    time_period_names = {0: 'Early Morning', 1: 'Morning', 2: 'Afternoon', 3: 'Evening', 4: 'Night'}
    period_dist = df['time_period'].value_counts().sort_index()
    for period, count in period_dist.items():
        period_name = time_period_names.get(period, f'Unknown ({period})')
        print(f"  {period_name}: {count} samples ({count/len(df)*100:.1f}%)")
    
    print("\nMotion Detection Distribution:")
    motion_dist = df['motion_detected'].value_counts()
    print(f"  No Motion (0): {motion_dist.get(0, 0)} samples")
    print(f"  Motion (1): {motion_dist.get(1, 0)} samples")
    
    # Check for night + high ambient light scenario
    print("\n=== Scenario Analysis ===")
    night_mask = df['time_period'] == 4
    if night_mask.any():
        night_data = df[night_mask]
        ambient_75th = df['ambient_light'].quantile(0.75)
        high_ambient_night = night_data[night_data['ambient_light'] > ambient_75th]
        
        print(f"Night samples: {night_mask.sum()}")
        print(f"Night samples with high ambient light (>{ambient_75th:.2f}): {len(high_ambient_night)}")
        
        if len(high_ambient_night) == 0:
            print("⚠️  NO data for night + high ambient light scenario!")
            print("   → Data augmentation will be applied during training")
        else:
            print(f"   Average LED brightness in this scenario: {high_ambient_night['led_brightness'].mean():.2f}%")
    
    return df


def train_and_save_model(df, output_model_path='led_rf_model.pkl', 
                        n_estimators=100, max_depth=15, min_samples_split=5, 
                        min_samples_leaf=2):
    """
    Train Random Forest model and save it
    
    Args:
        df (pd.DataFrame): Training dataset
        output_model_path (str): Path to save trained model
        n_estimators (int): Number of trees in forest
        max_depth (int): Maximum tree depth
        min_samples_split (int): Minimum samples to split
        min_samples_leaf (int): Minimum samples in leaf
        
    Returns:
        tuple: (controller, metrics)
    """
    # Initialize controller
    controller = LEDBrightnessRFController()
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60 + "\n")
    
    metrics = controller.train(
        df, 
        test_size=0.2, 
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    
    # Save model
    print("\n" + "="*60)
    controller.save_model(output_model_path)
    
    # Save metrics to JSON
    metrics_path = output_model_path.replace('.pkl', '_metrics.json')
    metrics_to_save = {
        'test_mae': float(metrics['test_mae']),
        'test_rmse': float(metrics['test_rmse']),
        'test_r2': float(metrics['test_r2']),
        'train_mae': float(metrics['train_mae']),
        'train_r2': float(metrics['train_r2']),
        'feature_importance': metrics['feature_importance'].to_dict('records'),
        'model_params': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"✓ Metrics saved to: {metrics_path}")
    
    return controller, metrics


def test_predictions(controller):
    """
    Test predictions with various scenarios
    
    Args:
        controller: Trained LEDBrightnessRFController
    """
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60 + "\n")
    
    test_scenarios = [
        {
            'name': 'Night + High Ambient Light + Motion (Street Light)',
            'ambient_light': 800,
            'motion_detected': 1,
            'hour': 22,  # 10 PM
            'day_of_week': 1,  # Tuesday
        },
        {
            'name': 'Night + High Ambient Light + NO Motion (Street Light)',
            'ambient_light': 800,
            'motion_detected': 0,
            'hour': 22,  # 10 PM
            'day_of_week': 1,  # Tuesday
        },
        {
            'name': 'Night + Low Ambient Light + Motion (Dark)',
            'ambient_light': 50,
            'motion_detected': 1,
            'hour': 23,  # 11 PM
            'day_of_week': 4,  # Friday
        },
        {
            'name': 'Night + Low Ambient Light + NO Motion (Dark)',
            'ambient_light': 50,
            'motion_detected': 0,
            'hour': 2,  # 2 AM
            'day_of_week': 4,  # Friday
        },
        {
            'name': 'Morning + Medium Ambient Light (Weekday)',
            'ambient_light': 400,
            'motion_detected': 1,
            'hour': 8,  # 8 AM
            'day_of_week': 2,  # Wednesday
        },
        {
            'name': 'Morning + Medium Ambient Light (Weekend)',
            'ambient_light': 400,
            'motion_detected': 1,
            'hour': 8,  # 8 AM
            'day_of_week': 6,  # Sunday
        },
        {
            'name': 'Evening + Motion',
            'ambient_light': 200,
            'motion_detected': 1,
            'hour': 18,  # 6 PM
            'day_of_week': 0,  # Monday
        },
        {
            'name': 'Afternoon + High Ambient Light (Sunny)',
            'ambient_light': 900,
            'motion_detected': 0,
            'hour': 14,  # 2 PM
            'day_of_week': 3,  # Thursday
        },
        {
            'name': 'Early Morning + No Motion',
            'ambient_light': 100,
            'motion_detected': 0,
            'hour': 5,  # 5 AM
            'day_of_week': 5,  # Saturday
        },
        {
            'name': 'Evening + Low Light + Motion',
            'ambient_light': 150,
            'motion_detected': 1,
            'hour': 19,  # 7 PM
            'day_of_week': 2,  # Wednesday
        }
    ]
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    results = []
    for scenario in test_scenarios:
        sin_hour, cos_hour, time_period, _ = calculate_time_features(scenario['hour'])
        
        result = controller.predict(
            ambient_light=scenario['ambient_light'],
            motion_detected=scenario['motion_detected'],
            sin_hour=sin_hour,
            cos_hour=cos_hour,
            time_period=time_period,
            day_of_week=scenario['day_of_week']
        )
        
        print(f"Scenario: {scenario['name']}")
        print(f"  Inputs: {day_names[scenario['day_of_week']]} {scenario['hour']:02d}:00, Ambient={scenario['ambient_light']} lux, Motion={scenario['motion_detected']}")
        print(f"  → ML Predicted: {result['ml_predicted']}%")
        print(f"  → Final Brightness: {result['final_brightness']}%")
        print(f"  → Uncertainty: ±{result['uncertainty']}% (Confidence: {result['confidence']}%)")
        print()
        
        results.append({
            'scenario': scenario['name'],
            'inputs': scenario,
            'prediction': result
        })
    
    return results


def cross_validate_model(df, n_folds=5):
    """
    Perform cross-validation to assess model stability
    
    Args:
        df (pd.DataFrame): Dataset
        n_folds (int): Number of CV folds
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION")
    print("="*60 + "\n")
    
    feature_names = ['ambient_light', 'motion_detected', 'sin_hour', 
                    'cos_hour', 'time_period', 'day_of_week']
    
    X = df[feature_names]
    y = df['led_brightness']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    
    # Negative MAE scoring
    mae_scores = -cross_val_score(model, X, y, cv=n_folds, 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
    
    # R² scoring
    r2_scores = cross_val_score(model, X, y, cv=n_folds, 
                                scoring='r2', n_jobs=-1)
    
    print(f"Cross-Validation Results ({n_folds} folds):")
    print(f"\nMAE per fold:")
    for i, score in enumerate(mae_scores, 1):
        print(f"  Fold {i}: {score:.2f}%")
    print(f"\nMAE: {mae_scores.mean():.2f}% (±{mae_scores.std():.2f}%)")
    
    print(f"\nR² per fold:")
    for i, score in enumerate(r2_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    print(f"\nR²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LED Brightness Control Random Forest Model')
    parser.add_argument('--data', type=str, required=True, help='Path to training dataset (CSV)')
    parser.add_argument('--output', type=str, default='led_rf_model.pkl', help='Output model file path')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees in forest')
    parser.add_argument('--max-depth', type=int, default=15, help='Maximum tree depth')
    parser.add_argument('--min-samples-split', type=int, default=5, help='Minimum samples to split')
    parser.add_argument('--min-samples-leaf', type=int, default=2, help='Minimum samples in leaf')
    parser.add_argument('--cross-validate', action='store_true', help='Perform cross-validation')
    
    args = parser.parse_args()
    
    # Load and validate data
    df = load_and_validate_data(args.data)
    
    # Optional cross-validation
    if args.cross_validate:
        cross_validate_model(df, n_folds=5)
    
    # Train and save model
    controller, metrics = train_and_save_model(
        df, 
        output_model_path=args.output,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf
    )
    
    # Test predictions
    test_results = test_predictions(controller)
    
    # Save test results
    results_path = args.output.replace('.pkl', '_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"✓ Test results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📊 Model Performance:")
    print(f"   Test MAE: {metrics['test_mae']:.2f}%")
    print(f"   Test RMSE: {metrics['test_rmse']:.2f}%")
    print(f"   Test R²: {metrics['test_r2']:.4f}")
    print(f"\n💾 Saved Files:")
    print(f"   Model: {args.output}")
    print(f"   Metrics: {args.output.replace('.pkl', '_metrics.json')}")
    print(f"   Test Results: {results_path}")
    print(f"\n🚀 Next Steps:")
    print(f"   1. Convert model to C++ for ESP32:")
    print(f"      python convert_to_cpp.py --model {args.output}")
    print(f"   2. Upload to ESP32 and test with live sensors")
    print(f"   3. Collect user feedback for online learning")


if __name__ == "__main__":
    main()
