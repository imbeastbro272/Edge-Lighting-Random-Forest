"""
Generate synthetic training data for LED brightness control
Useful for creating larger datasets with specific scenarios
"""

import numpy as np
import pandas as pd
import argparse
from led_brightness_rf_model import calculate_time_features


def generate_brightness(ambient_light, motion_detected, hour, day_of_week, noise_level=0.05):
    """
    Generate realistic brightness values based on conditions
    
    Args:
        ambient_light (float): Ambient light in lux
        motion_detected (int): Motion detection (0 or 1)
        hour (int): Hour of day (0-23)
        day_of_week (int): Day of week (0-6)
        noise_level (float): Amount of random noise to add
        
    Returns:
        int: Brightness percentage (0-100)
    """
    # Base brightness from ambient light (inverse relationship)
    base_brightness = 100 - (ambient_light / 10)
    base_brightness = np.clip(base_brightness, 0, 100)
    
    # Time-based adjustments
    if 0 <= hour < 6:  # Early morning
        time_multiplier = 0.5
    elif 6 <= hour < 12:  # Morning
        time_multiplier = 0.7
    elif 12 <= hour < 17:  # Afternoon
        time_multiplier = 0.8
    elif 17 <= hour < 21:  # Evening
        time_multiplier = 1.0
    else:  # Night
        time_multiplier = 0.6
    
    # Motion adjustment
    motion_boost = 25 if motion_detected else 0
    
    # Weekend adjustment (slightly lower brightness)
    weekend_factor = 0.9 if day_of_week >= 5 else 1.0
    
    # Calculate final brightness
    brightness = (base_brightness * time_multiplier * weekend_factor) + motion_boost
    
    # Add noise
    noise = np.random.normal(0, brightness * noise_level)
    brightness += noise
    
    # Clip to valid range
    brightness = np.clip(brightness, 0, 100)
    
    return int(round(brightness))


def generate_dataset(n_samples=1000, seed=42):
    """
    Generate synthetic dataset
    
    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated dataset
    """
    np.random.seed(seed)
    
    print(f"\nGenerating {n_samples} synthetic samples...")
    
    data = []
    
    for _ in range(n_samples):
        # Random conditions
        hour = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        motion_detected = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% motion
        
        # Ambient light varies by time of day
        if 0 <= hour < 6:  # Night/Early morning
            ambient_light = np.random.uniform(20, 200)
        elif 6 <= hour < 12:  # Morning
            ambient_light = np.random.uniform(200, 500)
        elif 12 <= hour < 17:  # Afternoon
            ambient_light = np.random.uniform(500, 1000)
        elif 17 <= hour < 21:  # Evening
            ambient_light = np.random.uniform(200, 600)
        else:  # Night
            ambient_light = np.random.uniform(50, 300)
        
        # Add occasional high ambient light at night (street lights, moon)
        if hour >= 21 or hour < 6:
            if np.random.random() < 0.15:  # 15% chance
                ambient_light = np.random.uniform(600, 900)
        
        # Calculate time features
        sin_hour, cos_hour, time_period, _ = calculate_time_features(hour)
        
        # Generate brightness
        brightness = generate_brightness(ambient_light, motion_detected, hour, day_of_week)
        
        data.append({
            'ambient_light': round(ambient_light, 2),
            'motion_detected': motion_detected,
            'sin_hour': round(sin_hour, 4),
            'cos_hour': round(cos_hour, 4),
            'time_period': time_period,
            'day_of_week': day_of_week,
            'led_brightness': brightness
        })
    
    df = pd.DataFrame(data)
    
    # Statistics
    print("\n=== Generated Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"\nBrightness range: {df['led_brightness'].min()}-{df['led_brightness'].max()}%")
    print(f"Brightness mean: {df['led_brightness'].mean():.2f}%")
    
    print("\nTime period distribution:")
    period_names = {0: 'Early Morning', 1: 'Morning', 2: 'Afternoon', 3: 'Evening', 4: 'Night'}
    for period, count in df['time_period'].value_counts().sort_index().items():
        print(f"  {period_names[period]}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\nMotion distribution:")
    print(f"  No motion: {(df['motion_detected']==0).sum()} ({(df['motion_detected']==0).sum()/len(df)*100:.1f}%)")
    print(f"  Motion: {(df['motion_detected']==1).sum()} ({(df['motion_detected']==1).sum()/len(df)*100:.1f}%)")
    
    print("\nDay of week distribution:")
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day, count in df['day_of_week'].value_counts().sort_index().items():
        print(f"  {day_names[day]}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def add_edge_cases(df, n_edge_cases=100):
    """
    Add specific edge cases to improve model robustness
    
    Args:
        df (pd.DataFrame): Existing dataset
        n_edge_cases (int): Number of edge cases to add
        
    Returns:
        pd.DataFrame: Dataset with edge cases
    """
    print(f"\nAdding {n_edge_cases} edge case samples...")
    
    edge_cases = []
    
    scenarios = [
        # Night + high ambient light scenarios
        {'hour': 22, 'ambient': (700, 900), 'motion': 1, 'brightness': (40, 60)},
        {'hour': 23, 'ambient': (700, 900), 'motion': 0, 'brightness': (15, 35)},
        {'hour': 0, 'ambient': (700, 900), 'motion': 1, 'brightness': (35, 55)},
        {'hour': 1, 'ambient': (700, 900), 'motion': 0, 'brightness': (10, 30)},
        
        # Very dark scenarios
        {'hour': 3, 'ambient': (0, 50), 'motion': 1, 'brightness': (80, 100)},
        {'hour': 4, 'ambient': (0, 50), 'motion': 0, 'brightness': (30, 50)},
        
        # Very bright daytime
        {'hour': 13, 'ambient': (900, 1000), 'motion': 1, 'brightness': (5, 15)},
        {'hour': 14, 'ambient': (900, 1000), 'motion': 0, 'brightness': (0, 10)},
        
        # Transition periods
        {'hour': 6, 'ambient': (100, 300), 'motion': 1, 'brightness': (40, 60)},
        {'hour': 18, 'ambient': (200, 400), 'motion': 1, 'brightness': (60, 80)},
    ]
    
    samples_per_scenario = n_edge_cases // len(scenarios)
    
    for scenario in scenarios:
        for _ in range(samples_per_scenario):
            hour = scenario['hour']
            ambient = np.random.uniform(*scenario['ambient'])
            motion = scenario['motion']
            brightness = np.random.randint(*scenario['brightness'])
            day_of_week = np.random.randint(0, 7)
            
            sin_hour, cos_hour, time_period, _ = calculate_time_features(hour)
            
            edge_cases.append({
                'ambient_light': round(ambient, 2),
                'motion_detected': motion,
                'sin_hour': round(sin_hour, 4),
                'cos_hour': round(cos_hour, 4),
                'time_period': time_period,
                'day_of_week': day_of_week,
                'led_brightness': brightness
            })
    
    edge_df = pd.DataFrame(edge_cases)
    combined_df = pd.concat([df, edge_df], ignore_index=True)
    
    print(f"✓ Added {len(edge_cases)} edge cases")
    print(f"Total samples: {len(combined_df)}")
    
    return combined_df


def main():
    """Main data generation pipeline"""
    parser = argparse.ArgumentParser(description='Generate synthetic training data for LED brightness control')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--edge-cases', type=int, default=100, help='Number of edge cases to add')
    parser.add_argument('--output', type=str, default='generated_training_data.csv', help='Output CSV file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no-edge-cases', action='store_true', help='Skip edge case generation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LED Brightness Training Data Generator")
    print("="*60)
    
    # Generate main dataset
    df = generate_dataset(n_samples=args.samples, seed=args.seed)
    
    # Add edge cases
    if not args.no_edge_cases:
        df = add_edge_cases(df, n_edge_cases=args.edge_cases)
    
    # Save to file
    df.to_csv(args.output, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Dataset saved to: {args.output}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Train model: python train_rf_model.py --data {args.output}")
    print(f"  2. Or combine with existing data:")
    print(f"     cat sample_led_data.csv {args.output} > combined_data.csv")
    print(f"     python train_rf_model.py --data combined_data.csv")


if __name__ == "__main__":
    main()
