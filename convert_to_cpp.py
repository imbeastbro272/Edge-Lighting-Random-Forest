"""
Convert trained Random Forest model to ESP32-compatible C++ code
Generates a header file with optimized decision tree implementations
"""

import pickle
import numpy as np
import argparse
from datetime import datetime


def extract_tree_structure(tree, feature_names):
    """
    Extract decision tree structure for C++ code generation
    
    Args:
        tree: Trained decision tree
        feature_names (list): List of feature names
        
    Returns:
        dict: Tree structure with nodes
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    
    nodes = []
    
    def recurse(node_id, depth=0):
        if tree_.feature[node_id] != -2:
            # Internal node
            feature = feature_name[node_id]
            threshold = tree_.threshold[node_id]
            left_child = tree_.children_left[node_id]
            right_child = tree_.children_right[node_id]
            
            nodes.append({
                'id': node_id,
                'type': 'internal',
                'feature': feature,
                'threshold': threshold,
                'left': left_child,
                'right': right_child,
                'depth': depth
            })
            
            recurse(left_child, depth + 1)
            recurse(right_child, depth + 1)
        else:
            # Leaf node
            value = tree_.value[node_id][0][0]
            nodes.append({
                'id': node_id,
                'type': 'leaf',
                'value': value,
                'depth': depth
            })
    
    recurse(0)
    return nodes


def generate_tree_cpp_code(tree_id, nodes, feature_names):
    """
    Generate C++ code for a single decision tree
    
    Args:
        tree_id (int): Tree index
        nodes (list): List of node dictionaries
        feature_names (list): Feature names
        
    Returns:
        str: C++ function code
    """
    feature_indices = {name: idx for idx, name in enumerate(feature_names)}
    
    code = f"// Decision Tree {tree_id}\n"
    code += f"float tree_{tree_id}(float features[]) {{\n"
    
    def generate_node_code(node_id, indent=1):
        node = next(n for n in nodes if n['id'] == node_id)
        indent_str = "  " * indent
        
        if node['type'] == 'leaf':
            return f"{indent_str}return {node['value']:.4f};\n"
        else:
            feature_idx = feature_indices[node['feature']]
            threshold = node['threshold']
            
            code = f"{indent_str}if (features[{feature_idx}] <= {threshold:.4f}f) {{\n"
            code += generate_node_code(node['left'], indent + 1)
            code += f"{indent_str}}} else {{\n"
            code += generate_node_code(node['right'], indent + 1)
            code += f"{indent_str}}}\n"
            return code
    
    code += generate_node_code(0)
    code += "}\n\n"
    
    return code


def generate_cpp_header(model_data, output_path='led_rf_model.h', max_trees=50):
    """
    Generate complete C++ header file with Random Forest model
    
    Args:
        model_data (dict): Loaded model data
        output_path (str): Output header file path
        max_trees (int): Maximum number of trees to export (for ESP32 memory limits)
    """
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    n_trees = min(len(model.estimators_), max_trees)
    
    print(f"\n{'='*60}")
    print(f"CONVERTING RANDOM FOREST TO C++")
    print(f"{'='*60}\n")
    print(f"Total trees in model: {len(model.estimators_)}")
    print(f"Exporting: {n_trees} trees (ESP32 memory limit)")
    print(f"Features: {feature_names}")
    
    # Start building header file
    header_code = f"""/*
 * LED Brightness Random Forest Model
 * Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * 
 * Model Information:
 * - Trees: {n_trees}
 * - Features: {len(feature_names)}
 * - Input features: {', '.join(feature_names)}
 * 
 * Feature Order (index):
 * 0: ambient_light (lux)
 * 1: motion_detected (0 or 1)
 * 2: sin_hour (sine of hour)
 * 3: cos_hour (cosine of hour)
 * 4: time_period (0-4)
 * 5: day_of_week (0-6)
 */

#ifndef LED_RF_MODEL_H
#define LED_RF_MODEL_H

#include <Arduino.h>
#include <math.h>

// Model Configuration
#define N_TREES {n_trees}
#define N_FEATURES {len(feature_names)}

// Feature indices
#define FEAT_AMBIENT_LIGHT 0
#define FEAT_MOTION_DETECTED 1
#define FEAT_SIN_HOUR 2
#define FEAT_COS_HOUR 3
#define FEAT_TIME_PERIOD 4
#define FEAT_DAY_OF_WEEK 5

"""
    
    # Generate tree functions
    print("\nGenerating tree functions...")
    tree_functions = ""
    
    for i, estimator in enumerate(model.estimators_[:n_trees]):
        if i % 10 == 0:
            print(f"  Processing tree {i+1}/{n_trees}...")
        
        nodes = extract_tree_structure(estimator, feature_names)
        tree_functions += generate_tree_cpp_code(i, nodes, feature_names)
    
    header_code += tree_functions
    
    # Generate prediction function
    header_code += f"""// Random Forest Prediction
float predict_rf(float features[N_FEATURES]) {{
  float sum = 0.0f;
  
  // Sum predictions from all trees
"""
    
    for i in range(n_trees):
        header_code += f"  sum += tree_{i}(features);\n"
    
    header_code += f"""  
  // Average predictions
  float prediction = sum / {n_trees}.0f;
  
  // Clip to valid range [0, 100]
  if (prediction < 0.0f) prediction = 0.0f;
  if (prediction > 100.0f) prediction = 100.0f;
  
  return prediction;
}}

// Helper: Calculate time features from hour
void calculate_time_features(int hour, float* sin_hour, float* cos_hour, int* time_period) {{
  // Cyclic encoding for hour
  *sin_hour = sin(2.0f * PI * hour / 24.0f);
  *cos_hour = cos(2.0f * PI * hour / 24.0f);
  
  // Time period encoding
  if (hour < 6) {{
    *time_period = 0;  // Early Morning
  }} else if (hour < 12) {{
    *time_period = 1;  // Morning
  }} else if (hour < 17) {{
    *time_period = 2;  // Afternoon
  }} else if (hour < 21) {{
    *time_period = 3;  // Evening
  }} else {{
    *time_period = 4;  // Night
  }}
}}

// High-level prediction function
int predict_led_brightness(float ambient_light, int motion_detected, int hour, int day_of_week) {{
  float features[N_FEATURES];
  float sin_hour, cos_hour;
  int time_period;
  
  // Calculate time features
  calculate_time_features(hour, &sin_hour, &cos_hour, &time_period);
  
  // Prepare feature array
  features[FEAT_AMBIENT_LIGHT] = ambient_light;
  features[FEAT_MOTION_DETECTED] = (float)motion_detected;
  features[FEAT_SIN_HOUR] = sin_hour;
  features[FEAT_COS_HOUR] = cos_hour;
  features[FEAT_TIME_PERIOD] = (float)time_period;
  features[FEAT_DAY_OF_WEEK] = (float)day_of_week;
  
  // Get prediction
  float prediction = predict_rf(features);
  
  // Return as integer percentage
  return (int)round(prediction);
}}

// Get current day of week (0=Sunday in RTC, convert to 0=Monday)
int get_day_of_week_from_rtc(int rtc_day_of_week) {{
  // RTC: 0=Sunday, 1=Monday, ..., 6=Saturday
  // Model: 0=Monday, 1=Tuesday, ..., 6=Sunday
  if (rtc_day_of_week == 0) return 6;  // Sunday
  return rtc_day_of_week - 1;
}}

#endif // LED_RF_MODEL_H
"""
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(header_code)
    
    # Calculate file size
    import os
    file_size_kb = os.path.getsize(output_path) / 1024
    
    print(f"\n{'='*60}")
    print(f"✓ C++ header generated successfully!")
    print(f"{'='*60}")
    print(f"\nOutput file: {output_path}")
    print(f"File size: {file_size_kb:.2f} KB")
    print(f"\nMemory estimate for ESP32:")
    print(f"  Flash (PROGMEM): ~{file_size_kb:.0f} KB")
    print(f"  RAM (runtime): ~{n_trees * 0.5:.1f} KB")
    
    if file_size_kb > 200:
        print(f"\n⚠️  Warning: Large model size ({file_size_kb:.0f} KB)")
        print(f"   Consider reducing --max-trees to fit ESP32 flash memory")
    
    return output_path


def generate_cpp_test_code(output_path='test_model.cpp'):
    """
    Generate test C++ code to verify model
    
    Args:
        output_path (str): Output test file path
    """
    test_code = """/*
 * Test file for LED Random Forest Model
 * Compile with: g++ -o test_model test_model.cpp -lm
 */

#include <iostream>
#include <iomanip>
#include "led_rf_model.h"

int main() {
  std::cout << "Testing LED Brightness Random Forest Model\\n";
  std::cout << "==========================================\\n\\n";
  
  // Test scenarios
  struct TestCase {
    const char* name;
    float ambient_light;
    int motion_detected;
    int hour;
    int day_of_week;
  };
  
  TestCase tests[] = {
    {"Night + High Ambient + Motion", 800.0f, 1, 22, 1},
    {"Night + High Ambient + No Motion", 800.0f, 0, 22, 1},
    {"Night + Low Ambient + Motion", 50.0f, 1, 23, 4},
    {"Morning + Medium Ambient", 400.0f, 1, 8, 2},
    {"Evening + Motion", 200.0f, 1, 18, 0},
    {"Afternoon + High Ambient", 900.0f, 0, 14, 3}
  };
  
  int n_tests = sizeof(tests) / sizeof(TestCase);
  
  for (int i = 0; i < n_tests; i++) {
    TestCase& test = tests[i];
    
    int brightness = predict_led_brightness(
      test.ambient_light,
      test.motion_detected,
      test.hour,
      test.day_of_week
    );
    
    std::cout << "Test: " << test.name << "\\n";
    std::cout << "  Ambient: " << test.ambient_light << " lux, ";
    std::cout << "Motion: " << test.motion_detected << ", ";
    std::cout << "Hour: " << test.hour << "\\n";
    std::cout << "  → Predicted Brightness: " << brightness << "%\\n\\n";
  }
  
  return 0;
}
"""
    
    with open(output_path, 'w') as f:
        f.write(test_code)
    
    print(f"\n✓ Test code generated: {output_path}")
    print(f"  Compile with: g++ -o test_model {output_path} -lm")


def main():
    """Main conversion pipeline"""
    parser = argparse.ArgumentParser(description='Convert Random Forest model to ESP32 C++ code')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pkl)')
    parser.add_argument('--output', type=str, default='led_rf_model.h', help='Output C++ header file')
    parser.add_argument('--max-trees', type=int, default=50, help='Maximum trees to export (ESP32 limit)')
    parser.add_argument('--generate-test', action='store_true', help='Generate test C++ code')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    with open(args.model, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✓ Model loaded successfully")
    print(f"  Trees in model: {len(model_data['model'].estimators_)}")
    
    # Generate C++ header
    header_path = generate_cpp_header(model_data, args.output, args.max_trees)
    
    # Optionally generate test code
    if args.generate_test:
        test_path = args.output.replace('.h', '_test.cpp')
        generate_cpp_test_code(test_path)
    
    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📁 Generated Files:")
    print(f"   {header_path}")
    if args.generate_test:
        print(f"   {test_path}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Copy {header_path} to your ESP32 Arduino project")
    print(f"   2. Include in your sketch: #include \"{header_path}\"")
    print(f"   3. Use: predict_led_brightness(ambient, motion, hour, day)")
    print(f"   4. Upload to ESP32 and test with live sensors!")


if __name__ == "__main__":
    main()
