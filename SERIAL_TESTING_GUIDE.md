# 📟 Serial Monitor Testing Guide

## Setup

1. **Upload Code to ESP32**
   - Open `esp32_adaptive_lighting.ino` in Arduino IDE
   - Select your ESP32 board
   - Upload the code

2. **Open Serial Monitor**
   - Go to **Tools → Serial Monitor** (or press `Ctrl+Shift+M`)
   - Set baud rate to **9600**
   - Set line ending to **"Both NL & CR"** or **"Newline"**

---

## 🎯 Basic Commands

### View All Commands
```
help
```
Shows complete list of available commands with syntax.

### View System Statistics
```
stats
```
**Output:**
```
=== STATISTICS ===
Predictions: 450
Average Brightness: 42.3
Pot Changes Today: 5
Training Samples: 23
Model Type: RANDOM FOREST
```

### View Model Information
```
modelinfo
```
Shows which model is active (Random Forest or Adaptive) and its parameters.

---

## 🧪 Manual Testing (WITHOUT Hardware)

### Test Command Format
```
test HH DD AMBIENT MOTION POT
```

**Parameters:**
- `HH` = Hour (0-23)
- `DD` = Day of week (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun)
- `AMBIENT` = Light level in lux (0-10000)
- `MOTION` = Motion detected (0=No, 1=Yes)
- `POT` = Potentiometer value (0-4095, where 2048=center/no offset)

### Example Test Scenarios

#### 1️⃣ **Night Time, Dark Room, No Motion**
```
test 22 1 50 0 2048
```
- 10 PM (22:00), Tuesday
- Very dark (50 lux)
- No motion
- No manual adjustment (pot at center)

**Expected:** High brightness (~70-90%)

---

#### 2️⃣ **Morning, Bright Light, Motion Detected**
```
test 9 2 5000 1 2048
```
- 9 AM, Wednesday
- Very bright (5000 lux)
- Motion detected
- No manual adjustment

**Expected:** Low brightness (~10-30%)

---

#### 3️⃣ **Evening, Medium Light, Motion, User Wants Brighter**
```
test 18 4 800 1 3000
```
- 6 PM, Friday
- Medium light (800 lux)
- Motion detected
- User wants it brighter (pot=3000, adds ~+40% brightness)

**Expected:** ML prediction + manual offset

---

#### 4️⃣ **Late Night, Dark, No Motion, User Wants Dimmer**
```
test 23 5 20 0 1500
```
- 11 PM, Saturday
- Very dark (20 lux)
- No motion
- User wants dimmer (pot=1500, adds ~-20% brightness)

**Expected:** ML prediction with negative offset

---

#### 5️⃣ **Midday, Bright, No Motion**
```
test 12 0 8000 0 2048
```
- Noon, Monday
- Very bright outside
- No motion
- No adjustment

**Expected:** Very low brightness (~5-15%)

---

## 📊 Sample Output

When you run a test, you'll see:

```
========================================
   MANUAL PREDICTION TEST RESULTS
========================================
Input Parameters:
  Time: 22:00 (Night)
  Day: Tuesday
  Ambient Light: 300 lux
  Motion: YES
  Pot Value: 2200 (Offset: +10%)

Model Predictions:
+------------------+----------+----------+----------+
| Model            | Raw (%)  | Offset   | Final(%) |
+------------------+----------+----------+----------+
| Random Forest    |   65.30 |      +10 |    75.30 |
| Adaptive(Learn)  |   68.50 |      +10 |    78.50 |
+------------------+----------+----------+----------+

Model Comparison:
  Difference: 3.2%
  Active Model: RANDOM FOREST (Original)
========================================
```

---

## 🎓 Training & Learning Commands

### View Training Samples
```
samples
```
Shows all captured training data from user adjustments.

### Clear All Training Data
```
clearsamples
```
⚠️ **Warning:** Deletes all learned data permanently!

### Force Retraining
```
retrain
```
Manually trigger model retraining with current samples (needs ≥5 samples).

### Reset Model to Defaults
```
resetmodel
```
Resets adaptive model and switches back to Random Forest.

---

## 🔍 Understanding Pot Values

The potentiometer controls manual brightness offset:

| Pot Value | Effect | Offset |
|-----------|--------|--------|
| 0 - 1848 | Decrease brightness | -100% to 0% |
| 1848 - 2248 | Deadzone (no change) | 0% |
| 2248 - 4095 | Increase brightness | 0% to +100% |

**Examples:**
- `pot 2048` = No offset (center)
- `pot 1000` = Reduce by ~40%
- `pot 3500` = Increase by ~60%

---

## 📝 Real-World Testing Scenarios

### Scenario 1: **Office During Work Hours**
```
test 10 2 3000 1 2048   # Morning meeting
test 14 2 4500 0 2048   # Afternoon, no one in room
test 17 2 2000 1 2048   # Evening, someone working late
```

### Scenario 2: **Home at Different Times**
```
test 7 6 100 1 2048     # Saturday morning, just woke up
test 12 6 6000 0 2048   # Midday, curtains open
test 20 6 500 1 2048    # Evening, watching TV
test 23 6 50 0 2048     # Going to bed
```

### Scenario 3: **Testing Manual Overrides**
```
test 21 0 200 0 2048    # ML prediction only
test 21 0 200 0 1500    # User wants dimmer
test 21 0 200 0 3000    # User wants brighter
```

---

## 🐛 Troubleshooting

**Problem:** No output when typing commands
- ✅ Check baud rate is **9600**
- ✅ Check line ending setting
- ✅ Try pressing Enter/Return after command

**Problem:** "Invalid format" error
- ✅ Check spacing between parameters
- ✅ Use format exactly: `test 22 1 300 1 2048` (spaces, not commas)

**Problem:** Model predictions seem off
- ✅ Check if adaptive model is active (type `modelinfo`)
- ✅ Try `resetmodel` to use original Random Forest
- ✅ Collect more training samples by using physical potentiometer

---

## 💡 Pro Tips

1. **Test Edge Cases**: Try extreme values (0 lux, 10000 lux) to see model behavior
2. **Compare Models**: After retraining, test same scenario to see how adaptive model differs
3. **Time of Day Patterns**: Test every hour (0-23) at same light level to see time patterns
4. **Day of Week**: Test same time on weekday vs weekend to check day patterns
5. **Motion Impact**: Test with/without motion (0 vs 1) to see motion weight

---

## 📞 Quick Reference Card

| Command | Purpose |
|---------|---------|
| `help` | Show all commands |
| `stats` | View statistics |
| `test HH DD AMBIENT MOTION POT` | Test prediction |
| `samples` | View training data |
| `retrain` | Force retraining |
| `modelinfo` | Check active model |
| `resetmodel` | Reset to defaults |
| `clearsamples` | Delete all training data |

---

## 🎉 Example Testing Session

```
> help
[shows all commands]

> stats
Predictions: 0
Training Samples: 0
Model Type: RANDOM FOREST

> test 22 1 300 1 2048
[shows prediction results]

> test 9 2 5000 0 2048
[shows different prediction for morning]

> modelinfo
Type: RANDOM FOREST (Original)
Update Count: 0

> stats
Predictions: 0
[note: test command doesn't increment predictions, only real-time loop does]
```

---

**Happy Testing! 🚀**

For issues or questions, check the main README or GitHub issues.
