# 🌟 ESP32 Adaptive LED Brightness Control with Random Forest ML

A complete machine learning system for adaptive LED brightness control on ESP32 microcontrollers using **Random Forest** regression with **live user feedback** and **online learning** capabilities.

## 🎯 Features

### Machine Learning
- **Random Forest Regressor** for robust predictions
- **Data augmentation** for better coverage of edge cases
- **Cross-validation** support for model evaluation
- **Feature importance** analysis
- **Uncertainty estimation** using tree variance

### ESP32 Integration
- **Live sensor input**: ambient light (LDR/BH1750) and motion (PIR)
- **Real-time predictions** using compiled C++ model
- **Web interface** for monitoring and control
- **User feedback collection** for continuous improvement
- **SPIFFS storage** for learning data persistence

### Online Learning
- **Feedback-based learning**: Collects user corrections
- **Learning buffer**: Stores up to 100 samples locally
- **Auto-save**: Periodically saves learning data
- **Export capability**: Download training data for model retraining

## 📁 Project Structure

```
Edge-Lighting-Random-Forest/
├── led_brightness_rf_model.py    # Random Forest model implementation
├── train_rf_model.py              # Training script with validation
├── convert_to_cpp.py              # Model to C++ converter
├── esp32_adaptive_lighting.ino    # ESP32 Arduino sketch
├── sample_led_data.csv            # Sample training dataset (250+ samples)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_rf_model.py --data sample_led_data.csv --output led_rf_model.pkl
```

**Optional parameters:**
- `--n-estimators 100`: Number of trees in forest (default: 100)
- `--max-depth 15`: Maximum tree depth (default: 15)
- `--min-samples-split 5`: Minimum samples to split (default: 5)
- `--min-samples-leaf 2`: Minimum samples in leaf (default: 2)
- `--cross-validate`: Perform 5-fold cross-validation

### 3. Convert Model to C++

```bash
python convert_to_cpp.py --model led_rf_model.pkl --output led_rf_model.h --max-trees 50
```

**Optional parameters:**
- `--max-trees 50`: Limit trees for ESP32 memory (default: 50)
- `--generate-test`: Generate C++ test file

### 4. Upload to ESP32

1. **Install Arduino IDE libraries:**
   - RTClib (by Adafruit)
   - ArduinoJson (version 6.x)

2. **Update WiFi credentials** in `esp32_adaptive_lighting.ino`:
   ```cpp
   const char* WIFI_SSID = "YourWiFiSSID";
   const char* WIFI_PASSWORD = "YourWiFiPassword";
   ```

3. **Configure pins** (if different from defaults):
   ```cpp
   #define PIN_LIGHT_SENSOR 34   // ADC pin for LDR
   #define PIN_MOTION_SENSOR 35  // Digital input for PIR
   #define PIN_LED_PWM 16        // PWM output for LED
   ```

4. **Copy** `led_rf_model.h` to your Arduino sketch folder

5. **Upload** to ESP32

### 5. Access Web Interface

1. Open Serial Monitor to find ESP32's IP address
2. Navigate to `http://<ESP32_IP>` in your browser
3. Monitor predictions and provide feedback

## 📊 Dataset Format

Your training data should be a CSV file with these columns:

| Column | Description | Range/Type |
|--------|-------------|------------|
| `ambient_light` | Ambient light level in lux | 0-1000+ float |
| `motion_detected` | Motion sensor status | 0 or 1 |
| `sin_hour` | Sine of hour (cyclic encoding) | -1.0 to 1.0 |
| `cos_hour` | Cosine of hour (cyclic encoding) | -1.0 to 1.0 |
| `time_period` | Time period of day | 0-4 integer |
| `day_of_week` | Day of week | 0-6 (Mon-Sun) |
| `led_brightness` | Target brightness percentage | 0-100 integer |

### Time Period Encoding
- `0`: Early Morning (00:00-05:59)
- `1`: Morning (06:00-11:59)
- `2`: Afternoon (12:00-16:59)
- `3`: Evening (17:00-20:59)
- `4`: Night (21:00-23:59)

## 🔧 Hardware Setup

### Required Components
- **ESP32** development board
- **LDR (Light Dependent Resistor)** or **BH1750** light sensor
- **PIR Motion Sensor** (HC-SR501 or similar)
- **LED strip** or high-power LED with MOSFET driver
- **DS3231 RTC module** (optional, can use NTP)
- **10kΩ resistor** (for LDR voltage divider)

### Wiring Diagram

```
ESP32 Pin Configuration:
┌─────────────────────────────────────┐
│                                     │
│  GPIO34 ──────────┬─── LDR          │
│                   │                 │
│                   └─── 10kΩ to GND  │
│                                     │
│  GPIO35 ─────────────── PIR (OUT)   │
│                                     │
│  GPIO16 ─────────────── LED (PWM)   │
│                                     │
│  GPIO21 (SDA) ───────── RTC SDA     │
│  GPIO22 (SCL) ───────── RTC SCL     │
│                                     │
│  3.3V ───────────────── Sensors VCC │
│  GND ────────────────── Sensors GND │
│                                     │
└─────────────────────────────────────┘
```

### LED Driver Circuit (for high-power LEDs)

```
GPIO16 ──┬─── 1kΩ ───┬─── MOSFET Gate (IRLZ44N)
         │           │
         └─ 10kΩ ────┴─── GND

MOSFET Drain ──── LED Strip (-)
MOSFET Source ─── GND
LED Strip (+) ──── 12V Power Supply (+)
ESP32 GND ────────  12V Power Supply (-)
```

## 🌐 Web Interface

The ESP32 hosts a web interface at `http://<ESP32_IP>` with the following features:

### Dashboard
- **Current Status**: Ambient light, motion, time, predicted/actual brightness
- **Feedback Controls**: Accept or adjust predictions
- **Manual Control**: Slider to set brightness manually
- **Statistics**: Predictions made, feedbacks received, learning buffer status
- **Export**: Download collected learning data

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/status` | GET | Current system status (JSON) |
| `/api/feedback` | POST | Submit feedback (accept/reject) |
| `/api/manual` | POST | Set brightness manually |
| `/api/stats` | GET | System statistics (JSON) |
| `/api/export` | GET | Download learning data (CSV) |

## 🎓 Model Training Details

### Random Forest Configuration

**Default hyperparameters:**
- **n_estimators**: 100 trees (configurable)
- **max_depth**: 15 levels (prevents overfitting)
- **min_samples_split**: 5 samples (balance between bias/variance)
- **min_samples_leaf**: 2 samples (smoother predictions)

### Why Random Forest?

✅ **Advantages over Decision Tree:**
1. **Better generalization**: Reduces overfitting by averaging multiple trees
2. **Robust to noise**: Less sensitive to outliers in training data
3. **Uncertainty estimation**: Tree variance indicates prediction confidence
4. **Feature importance**: Identifies most influential factors
5. **Non-linear relationships**: Captures complex interactions

### Performance Metrics

The training script reports:
- **MAE (Mean Absolute Error)**: Average prediction error in %
- **RMSE (Root Mean Squared Error)**: Penalizes large errors
- **R² Score**: Proportion of variance explained (closer to 1.0 is better)
- **Feature Importance**: Which inputs matter most

**Target Performance:**
- Test MAE: < 5%
- Test R²: > 0.85

## 🔄 Online Learning Workflow

1. **Prediction**: Model predicts brightness based on sensors
2. **User Feedback**:
   - **Accept**: Prediction was good → add to learning buffer
   - **Adjust**: User sets different brightness → add correction to buffer
   - **Timeout**: No feedback after 30s → assume prediction was acceptable
3. **Buffer Management**: Stores up to 100 recent samples in SPIFFS
4. **Export**: User downloads accumulated data
5. **Retrain**: Run training script with new data
6. **Deploy**: Convert updated model to C++ and upload to ESP32

## 📈 Advanced Usage

### Custom Dataset

Create your own dataset by:
1. Collecting sensor readings over time
2. Manually setting desired brightness for each condition
3. Saving in CSV format with required columns
4. Training model with your data

### Model Optimization

**For better accuracy:**
```bash
python train_rf_model.py \
  --data your_data.csv \
  --n-estimators 150 \
  --max-depth 20 \
  --cross-validate
```

**For smaller ESP32 memory footprint:**
```bash
python convert_to_cpp.py \
  --model led_rf_model.pkl \
  --max-trees 30 \
  --output led_rf_model.h
```

### Continuous Learning Loop

```bash
# 1. Initial training
python train_rf_model.py --data sample_led_data.csv --output model_v1.pkl

# 2. Deploy to ESP32
python convert_to_cpp.py --model model_v1.pkl --output led_rf_model.h

# 3. Collect feedback (via web interface)
# Download learning data from http://<ESP32_IP>/api/export

# 4. Combine with original data
cat sample_led_data.csv export_training_data.csv > combined_data.csv

# 5. Retrain with feedback
python train_rf_model.py --data combined_data.csv --output model_v2.pkl

# 6. Repeat cycle
```

## 🐛 Troubleshooting

### Model Training Issues

**Problem**: High test MAE (> 10%)
- **Solution**: Collect more diverse training data, especially edge cases
- Try increasing `--n-estimators` or `--max-depth`

**Problem**: Model file too large
- **Solution**: Reduce `--max-trees` when converting to C++
- Use smaller `--max-depth` during training

### ESP32 Issues

**Problem**: WiFi won't connect
- **Solution**: Check credentials, ensure 2.4GHz WiFi (ESP32 doesn't support 5GHz)

**Problem**: Web interface not loading
- **Solution**: Check Serial Monitor for IP address, verify ESP32 is on same network

**Problem**: Compile error "led_rf_model.h not found"
- **Solution**: Copy header file to Arduino sketch directory

**Problem**: Out of memory error
- **Solution**: Reduce `--max-trees` (try 30 instead of 50)

### Sensor Issues

**Problem**: Light readings always 0 or 4095
- **Solution**: Check LDR wiring, verify voltage divider with 10kΩ resistor

**Problem**: Motion sensor always triggered
- **Solution**: Adjust PIR sensitivity potentiometer, check for heat sources

## 📊 Performance Comparison

### Decision Tree vs Random Forest

| Metric | Decision Tree | Random Forest |
|--------|---------------|---------------|
| Test MAE | ~6-8% | ~3-5% |
| Test R² | 0.75-0.82 | 0.85-0.92 |
| Overfitting Risk | High | Low |
| Generalization | Moderate | Excellent |
| Model Size | Small (~50KB) | Larger (~150KB) |
| Inference Time | Very Fast | Fast |

## 🔒 Memory Usage (ESP32)

| Component | Flash (PROGMEM) | RAM (Runtime) |
|-----------|-----------------|---------------|
| Model (50 trees) | ~150 KB | ~25 KB |
| Web Server | ~30 KB | ~15 KB |
| SPIFFS (learning buffer) | 256 KB | ~5 KB |
| Arduino Core | ~100 KB | ~40 KB |
| **Total** | **~536 KB / 4 MB** | **~85 KB / 520 KB** |

✅ **Fits comfortably** in ESP32 resources with room for expansion!

## 🌟 Future Enhancements

- [ ] Support for multiple LED zones
- [ ] Integration with smart home systems (MQTT, HomeKit)
- [ ] Advanced sensors (temperature, humidity)
- [ ] Seasonal learning (adjust for daylight changes)
- [ ] Mobile app for easier feedback
- [ ] OTA (Over-The-Air) model updates
- [ ] Energy consumption tracking
- [ ] Anomaly detection for sensor failures

## 📚 References

- **Scikit-learn Random Forest**: https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
- **ESP32 Documentation**: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/
- **Arduino JSON**: https://arduinojson.org/
- **RTClib**: https://github.com/adafruit/RTClib

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Share your trained models and datasets

## 📝 License

This project is open-source and available under the MIT License.

## 👨‍💻 Author

Created for adaptive smart lighting applications with machine learning on resource-constrained devices.

---

## ⚡ Quick Command Reference

```bash
# Train model
python train_rf_model.py --data sample_led_data.csv --output led_rf_model.pkl

# With cross-validation
python train_rf_model.py --data sample_led_data.csv --cross-validate

# Convert to C++
python convert_to_cpp.py --model led_rf_model.pkl --output led_rf_model.h

# Generate test code
python convert_to_cpp.py --model led_rf_model.pkl --generate-test

# Smaller model for ESP32
python convert_to_cpp.py --model led_rf_model.pkl --max-trees 30
```

---

**Happy Building! 🚀💡**
