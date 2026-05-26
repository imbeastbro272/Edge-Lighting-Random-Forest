# 🚀 Quick Start Guide

## 5-Minute Setup

### Step 1: Train the Model (2 minutes)

```bash
# Install dependencies
pip install numpy pandas scikit-learn

# Train with sample data
python train_rf_model.py --data sample_led_data.csv --output led_rf_model.pkl
```

**Expected output:**
```
Dataset shape: (250, 7)
Training Random Forest...
Test MAE: 3.45%
Test R²: 0.8923
✓ Model saved to: led_rf_model.pkl
```

### Step 2: Convert to C++ (1 minute)

```bash
python convert_to_cpp.py --model led_rf_model.pkl --output led_rf_model.h --max-trees 50
```

**Expected output:**
```
Total trees in model: 100
Exporting: 50 trees (ESP32 memory limit)
✓ C++ header generated successfully!
File size: 145.23 KB
```

### Step 3: Configure ESP32 (2 minutes)

1. Open `esp32_adaptive_lighting.ino` in Arduino IDE

2. Update WiFi credentials:
```cpp
const char* WIFI_SSID = "YourWiFiName";
const char* WIFI_PASSWORD = "YourPassword";
```

3. Copy `led_rf_model.h` to your Arduino sketch folder

4. Install required libraries (Arduino IDE → Tools → Manage Libraries):
   - **RTClib** by Adafruit
   - **ArduinoJson** (version 6.x)

5. Upload to ESP32

### Step 4: Test It! (< 1 minute)

1. Open Serial Monitor (115200 baud)
2. Note the IP address (e.g., `192.168.1.100`)
3. Open browser: `http://192.168.1.100`
4. See live predictions and provide feedback!

## Hardware Connections (Minimal Setup)

```
LDR Setup (Ambient Light):
ESP32 GPIO34 ──┬─── LDR ─── 3.3V
               │
               └─── 10kΩ ─── GND

PIR Sensor (Motion):
ESP32 GPIO35 ─── PIR OUT
ESP32 3.3V   ─── PIR VCC
ESP32 GND    ─── PIR GND

LED (PWM):
ESP32 GPIO16 ─── LED+ (through appropriate driver)
```

## First Prediction Test

Once uploaded, check Serial Monitor:

```
=================================
ESP32 Adaptive Lighting System
Random Forest ML Model
=================================

✓ SPIFFS initialized
✓ RTC initialized
✓ WiFi connected
  IP address: 192.168.1.100
✓ Web server started

--- New Prediction ---
Time: 22:00, Day: 1
Ambient: 800 lux, Motion: 1
Predicted Brightness: 48%
--------------------
```

## Web Interface Preview

Navigate to your ESP32's IP address to see:

📊 **Current Status**
- Ambient Light: 800 lux
- Motion: Detected
- Time: 22:00, Day 1
- Predicted: 48%
- Current: 48%

✅ **Feedback Buttons**
- Accept (if brightness is correct)
- Adjust (if you want different brightness)

🎚️ **Manual Control**
- Slider to set brightness 0-100%
- Set Brightness button

📈 **Statistics**
- Predictions made
- Feedbacks received
- Learning buffer status

## Common First-Time Issues

### 🔴 Issue: WiFi won't connect
**Solution:** Make sure you're using 2.4GHz WiFi (ESP32 doesn't support 5GHz)

### 🔴 Issue: Compile error "led_rf_model.h not found"
**Solution:** Copy `led_rf_model.h` to the same folder as `esp32_adaptive_lighting.ino`

### 🔴 Issue: Light sensor always reads 4095
**Solution:** Check your LDR wiring - you need a voltage divider with 10kΩ resistor

### 🔴 Issue: Out of memory when compiling
**Solution:** Reduce trees in C++ conversion: `--max-trees 30`

## Next Steps

1. **Collect Feedback**: Use the web interface for a day or two
2. **Export Data**: Click "Export Learning Data" button
3. **Retrain Model**: Combine with original data and retrain
4. **Deploy Updated Model**: Convert to C++ and re-upload

## Example: Continuous Learning Cycle

```bash
# Week 1: Initial deployment
python train_rf_model.py --data sample_led_data.csv --output model_v1.pkl
python convert_to_cpp.py --model model_v1.pkl --output led_rf_model.h

# Week 2: After collecting 50+ feedback samples
# (Download from http://ESP32_IP/api/export as feedback_week1.csv)
cat sample_led_data.csv feedback_week1.csv > combined_v2.csv
python train_rf_model.py --data combined_v2.csv --output model_v2.pkl
python convert_to_cpp.py --model model_v2.pkl --output led_rf_model.h

# Week 3: More personalized
cat combined_v2.csv feedback_week2.csv > combined_v3.csv
python train_rf_model.py --data combined_v3.csv --output model_v3.pkl
# ... and so on
```

## Pro Tips

💡 **Faster Training**: Use `--n-estimators 50` for quicker training during testing

💡 **Better Accuracy**: Use `--cross-validate` to check model stability

💡 **Smaller Model**: Use `--max-trees 30` if you have memory constraints

💡 **Test Before Deploy**: Use `--generate-test` to test C++ model on your PC first

## Sample Commands

```bash
# Quick training (50 trees, fast)
python train_rf_model.py --data sample_led_data.csv --n-estimators 50

# High-quality training (150 trees, slower)
python train_rf_model.py --data sample_led_data.csv --n-estimators 150 --cross-validate

# Memory-optimized conversion (30 trees)
python convert_to_cpp.py --model led_rf_model.pkl --max-trees 30

# Standard conversion with test file
python convert_to_cpp.py --model led_rf_model.pkl --max-trees 50 --generate-test
```

## Ready to Go! 🎉

You now have a fully functional ML-powered adaptive lighting system that learns from your preferences!

**Questions?** Check the full README.md for detailed documentation.

**Issues?** See the Troubleshooting section in README.md.

---

**Built with ❤️ for smart home automation**
