/*
 * ESP32 Adaptive LED Brightness Control with Random Forest
 * 
 * Features:
 * - Live sensor input (ambient light, motion)
 * - Random Forest ML-based brightness prediction
 * - User feedback collection for online learning
 * - EEPROM storage for learning data
 * - Serial interface for monitoring and control
 * - Manual control via potentiometer
 * 
 * Hardware Requirements:
 * - ESP32 board
 * - LDR (Light Dependent Resistor)
 * - PIR motion sensor
 * - LED strip (PWM controlled)
 * - RTC module (DS3231)
 * - Potentiometer for manual adjustment
 * 
 * Pin Configuration:
 * - GPIO34: LDR/Light sensor (ADC)
 * - GPIO27: PIR motion sensor
 * - GPIO33: Potentiometer (manual control)
 * - GPIO2: LED PWM output
 */

#include <Wire.h>
#include <RTClib.h>
#include <EEPROM.h>
#include <math.h>
#include "led_rf_model.h"  // Generated Random Forest model header

// ============================================================================
// PIN DEFINITIONS
// ============================================================================

#define PIR_PIN 27
#define LDR_PIN 34
#define POT_PIN 33
#define LED_PIN 2

// ============================================================================
// CONFIGURATION
// ============================================================================

#define UPDATE_INTERVAL 1000
#define SMOOTHING_FACTOR 0.7
#define POT_DEADZONE 200

#define LEARNING_ENABLED true

#define POT_STABLE_DURATION 5000
#define POT_CHANGE_THRESHOLD 50
#define MAX_CHANGES_BEFORE_RETRAIN 20

#define RETRAIN_HOUR 0
#define RETRAIN_MINUTE 0
#define RETRAIN_SECOND 1

#define EEPROM_SIZE 4096

// ============================================================================
// TRAINING SAMPLE STORAGE
// ============================================================================

#define MAX_TRAINING_SAMPLES 100
#define SAMPLE_SIZE 32  // bytes per sample
#define EEPROM_SAMPLES_START 100
#define EEPROM_SAMPLE_COUNT_ADDR 0
#define EEPROM_MODEL_WEIGHTS_START 3300  // Store model weights here

struct TrainingSample {
  float ambient_light;
  float sin_hour;
  float cos_hour;
  int motion_detected;
  int time_period;
  int day_of_week;
  float target_brightness;  // User's desired brightness
  uint32_t timestamp;
};

int training_sample_count = 0;

// ============================================================================
// ADAPTIVE MODEL PARAMETERS
// ============================================================================

// Simplified adaptive model: weighted features + bias
struct ModelWeights {
  float ambient_weight;
  float motion_weight;
  float sin_hour_weight;
  float cos_hour_weight;
  float time_period_weights[5];  // 5 time periods
  float day_of_week_weights[7];  // 7 days
  float bias;
  float learning_rate;
  uint32_t update_count;
};

ModelWeights adaptive_model;
bool use_adaptive_model = false;  // Start with RF, switch after retraining

// ============================================================================
// RTC
// ============================================================================

RTC_DS3231 rtc;

// ============================================================================
// STATE VARIABLES
// ============================================================================

float smoothed_ldr = 0;
float smoothed_brightness = 0;
int manual_offset = 0;
unsigned long last_update = 0;

// ============================================================================
// RETRAINING STATE
// ============================================================================

int last_stable_pot_value = -1;
int current_pot_candidate = -1;
unsigned long pot_stable_start = 0;
bool pot_candidate_active = false;

int pot_change_count_today = 0;
bool retrained_at_midnight = false;
bool retrained_at_20_changes = false;
int last_retrain_day = -1;

// ============================================================================
// STATS
// ============================================================================

unsigned long prediction_count = 0;
float avg_brightness = 0;

// ============================================================================
// TRAINING SAMPLE MANAGEMENT
// ============================================================================

void initializeTrainingStorage() {
  // Read sample count from EEPROM
  EEPROM.get(EEPROM_SAMPLE_COUNT_ADDR, training_sample_count);
  
  if (training_sample_count < 0 || training_sample_count > MAX_TRAINING_SAMPLES) {
    training_sample_count = 0;
    EEPROM.put(EEPROM_SAMPLE_COUNT_ADDR, training_sample_count);
    EEPROM.commit();
  }
  
  Serial.print(F("Training samples in memory: "));
  Serial.println(training_sample_count);
}

void saveTrainingSample(TrainingSample sample) {
  if (training_sample_count >= MAX_TRAINING_SAMPLES) {
    // Overwrite oldest sample (circular buffer)
    training_sample_count = 0;
  }
  
  int addr = EEPROM_SAMPLES_START + (training_sample_count * SAMPLE_SIZE);
  EEPROM.put(addr, sample);
  
  training_sample_count++;
  EEPROM.put(EEPROM_SAMPLE_COUNT_ADDR, training_sample_count);
  EEPROM.commit();
  
  Serial.print(F("Training sample saved. Total: "));
  Serial.println(training_sample_count);
}

TrainingSample loadTrainingSample(int index) {
  TrainingSample sample;
  int addr = EEPROM_SAMPLES_START + (index * SAMPLE_SIZE);
  EEPROM.get(addr, sample);
  return sample;
}

void captureTrainingSample(float ml_pred, float user_brightness) {
  DateTime now = rtc.now();
  
  TrainingSample sample;
  sample.ambient_light = smoothed_ldr;
  sample.sin_hour = sin(2.0 * PI * now.hour() / 24.0);
  sample.cos_hour = cos(2.0 * PI * now.hour() / 24.0);
  sample.motion_detected = digitalRead(PIR_PIN);
  sample.time_period = getTimePeriod(now.hour());
  sample.day_of_week = (now.dayOfTheWeek() + 6) % 7;
  sample.target_brightness = user_brightness;
  sample.timestamp = now.unixtime();
  
  saveTrainingSample(sample);
  
  Serial.print(F("Captured: Ambient="));
  Serial.print(sample.ambient_light);
  Serial.print(F(" ML="));
  Serial.print(ml_pred);
  Serial.print(F("% User="));
  Serial.print(user_brightness);
  Serial.println(F("%"));
}

// ============================================================================
// ADAPTIVE MODEL INITIALIZATION
// ============================================================================

void initializeAdaptiveModel() {
  // Initialize with reasonable defaults
  adaptive_model.ambient_weight = -0.02;  // More light = less brightness
  adaptive_model.motion_weight = 15.0;    // Motion adds brightness
  adaptive_model.sin_hour_weight = 5.0;
  adaptive_model.cos_hour_weight = 5.0;
  
  // Time period weights (early morning, morning, afternoon, evening, night)
  adaptive_model.time_period_weights[0] = 30.0;  // Early morning
  adaptive_model.time_period_weights[1] = 20.0;  // Morning
  adaptive_model.time_period_weights[2] = 15.0;  // Afternoon
  adaptive_model.time_period_weights[3] = 25.0;  // Evening
  adaptive_model.time_period_weights[4] = 40.0;  // Night
  
  // Day of week weights (Mon-Sun)
  for (int i = 0; i < 7; i++) {
    adaptive_model.day_of_week_weights[i] = 0.0;  // Neutral
  }
  
  adaptive_model.bias = 50.0;
  adaptive_model.learning_rate = 0.01;
  adaptive_model.update_count = 0;
}

void saveAdaptiveModel() {
  EEPROM.put(EEPROM_MODEL_WEIGHTS_START, adaptive_model);
  EEPROM.commit();
  Serial.println(F("Model weights saved to EEPROM"));
}

void loadAdaptiveModel() {
  EEPROM.get(EEPROM_MODEL_WEIGHTS_START, adaptive_model);
  
  // Validate loaded data
  if (isnan(adaptive_model.bias) || adaptive_model.update_count > 1000000) {
    Serial.println(F("Invalid model data, reinitializing"));
    initializeAdaptiveModel();
  } else {
    Serial.print(F("Loaded model with "));
    Serial.print(adaptive_model.update_count);
    Serial.println(F(" updates"));
  }
}

// ============================================================================
// PREDICTION WITH ADAPTIVE MODEL
// ============================================================================

float predictAdaptive(float ambient, int motion, float sin_h, float cos_h, 
                      int period, int day) {
  float prediction = adaptive_model.bias;
  
  prediction += adaptive_model.ambient_weight * ambient;
  prediction += adaptive_model.motion_weight * motion;
  prediction += adaptive_model.sin_hour_weight * sin_h;
  prediction += adaptive_model.cos_hour_weight * cos_h;
  prediction += adaptive_model.time_period_weights[period];
  prediction += adaptive_model.day_of_week_weights[day];
  
  return constrain(prediction, 0, 100);
}

// ============================================================================
// LIVE RETRAINING IMPLEMENTATION
// ============================================================================

void performRetraining() {
  Serial.println(F(""));
  Serial.println(F("========================================"));
  Serial.println(F("      LIVE RETRAINING IN PROGRESS"));
  Serial.println(F("========================================"));
  
  if (training_sample_count < 5) {
    Serial.println(F("Insufficient samples (need >=5). Skipping."));
    return;
  }
  
  Serial.print(F("Training on "));
  Serial.print(training_sample_count);
  Serial.println(F(" samples..."));
  
  // GRADIENT DESCENT RETRAINING
  int num_epochs = 10;
  float total_error = 0;
  
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    float epoch_error = 0;
    
    for (int i = 0; i < training_sample_count; i++) {
      TrainingSample sample = loadTrainingSample(i);
      
      // Forward pass
      float prediction = predictAdaptive(
        sample.ambient_light,
        sample.motion_detected,
        sample.sin_hour,
        sample.cos_hour,
        sample.time_period,
        sample.day_of_week
      );
      
      // Calculate error
      float error = sample.target_brightness - prediction;
      epoch_error += abs(error);
      
      // Backward pass - gradient descent
      float lr = adaptive_model.learning_rate;
      
      adaptive_model.ambient_weight += lr * error * sample.ambient_light;
      adaptive_model.motion_weight += lr * error * sample.motion_detected;
      adaptive_model.sin_hour_weight += lr * error * sample.sin_hour;
      adaptive_model.cos_hour_weight += lr * error * sample.cos_hour;
      adaptive_model.time_period_weights[sample.time_period] += lr * error;
      adaptive_model.day_of_week_weights[sample.day_of_week] += lr * error;
      adaptive_model.bias += lr * error;
    }
    
    if (epoch == num_epochs - 1) {
      total_error = epoch_error / training_sample_count;
    }
  }
  
  adaptive_model.update_count++;
  
  // Save updated model
  saveAdaptiveModel();
  
  // Enable adaptive model
  use_adaptive_model = true;
  
  Serial.println(F(""));
  Serial.println(F("RETRAINING COMPLETE"));
  Serial.print(F("  Average Error: "));
  Serial.print(total_error);
  Serial.println(F("%"));
  Serial.print(F("  Total Updates: "));
  Serial.println(adaptive_model.update_count);
  Serial.println(F("  Model: ADAPTIVE (User-Learned)"));
  Serial.println(F("========================================"));
  Serial.println(F(""));
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

float readLDR() {
  long sum = 0;
  
  for (int i = 0; i < 20; i++) {
    sum += analogRead(LDR_PIN);
    delay(2);
  }
  
  float avg = sum / 20.0;
  float voltage = (avg / 4095.0) * 3.3;
  
  if (voltage <= 0.01) {
    return 0;
  }
  
  float r_ldr = 10000.0 * voltage / (3.3 - voltage);
  float lux = 32768000.0 * pow(r_ldr, -1.4);
  lux = constrain(lux, 0, 100000);
  
  return lux;
}

int readManualOffset() {
  int pot_value = analogRead(POT_PIN);
  int center = 2048;
  
  if (pot_value >= center - POT_DEADZONE &&
      pot_value <= center + POT_DEADZONE) {
    return 0;
  }
  
  int offset;
  
  if (pot_value < center - POT_DEADZONE) {
    offset = map(pot_value, 0, center - POT_DEADZONE, -100, 0);
  } else {
    offset = map(pot_value, center + POT_DEADZONE, 4095, 0, 100);
  }
  
  return offset;
}

int getTimePeriod(int hour) {
  if (hour >= 4 && hour <= 6) {
    return 0;  // Early morning
  } else if (hour > 6 && hour <= 12) {
    return 1;  // Morning
  } else if (hour > 12 && hour <= 16) {
    return 2;  // Afternoon
  } else if (hour > 16 && hour <= 20) {
    return 3;  // Evening
  } else {
    return 4;  // Night
  }
}

void printDateTime(DateTime dt) {
  char buf[20];
  sprintf(buf, "%04d-%02d-%02d %02d:%02d:%02d",
          dt.year(), dt.month(), dt.day(),
          dt.hour(), dt.minute(), dt.second());
  Serial.print(buf);
}

// ============================================================================
// RETRAINING MONITORING
// ============================================================================

void monitorPotForRetraining(unsigned long current_time, 
                             float ml_pred, float user_brightness) {
  int current_pot_raw = analogRead(POT_PIN);
  
  if (pot_candidate_active) {
    if (abs(current_pot_raw - current_pot_candidate) <= POT_CHANGE_THRESHOLD) {
      if (current_time - pot_stable_start >= POT_STABLE_DURATION) {
        if (last_stable_pot_value == -1 ||
            abs(current_pot_raw - last_stable_pot_value) > POT_CHANGE_THRESHOLD) {
          
          last_stable_pot_value = current_pot_raw;
          pot_candidate_active = false;
          pot_change_count_today++;
          
          // CAPTURE TRAINING SAMPLE
          if (abs(manual_offset) > 5) {  // Only capture if user made significant adjustment
            captureTrainingSample(ml_pred, user_brightness);
          }
          
          Serial.print(F("Pot Change #"));
          Serial.println(pot_change_count_today);
          
          if (pot_change_count_today >= MAX_CHANGES_BEFORE_RETRAIN &&
              !retrained_at_20_changes) {
            performRetraining();
            retrained_at_20_changes = true;
          }
        }
      }
    } else {
      current_pot_candidate = current_pot_raw;
      pot_stable_start = current_time;
    }
  } else {
    if (last_stable_pot_value == -1) {
      last_stable_pot_value = current_pot_raw;
    } else if (abs(current_pot_raw - last_stable_pot_value) > POT_CHANGE_THRESHOLD) {
      pot_candidate_active = true;
      current_pot_candidate = current_pot_raw;
      pot_stable_start = current_time;
    }
  }
}

void checkScheduledRetraining(DateTime &now) {
  int current_day = now.day();
  
  if (current_day != last_retrain_day && last_retrain_day != -1) {
    pot_change_count_today = 0;
    retrained_at_midnight = false;
    retrained_at_20_changes = false;
    last_retrain_day = current_day;
  }
  
  if (last_retrain_day == -1) {
    last_retrain_day = current_day;
  }
  
  if (now.hour() == RETRAIN_HOUR &&
      now.minute() == RETRAIN_MINUTE &&
      now.second() == RETRAIN_SECOND &&
      !retrained_at_midnight) {
    performRetraining();
    retrained_at_midnight = true;
  }
}

// ============================================================================
// SERIAL COMMAND HANDLING
// ============================================================================

void handleSerialCommand() {
  String command = Serial.readStringUntil('\n');
  command.trim();
  
  if (command == "help") {
    Serial.println(F("=== COMMANDS ==="));
    Serial.println(F("help          - Show all commands"));
    Serial.println(F("stats         - Show statistics"));
    Serial.println(F("retrain       - Force retraining"));
    Serial.println(F("samples       - View training samples"));
    Serial.println(F("clearsamples  - Clear all samples"));
    Serial.println(F("modelinfo     - Show model summary"));
    Serial.println(F("resetmodel    - Reset model to defaults"));
  }
  else if (command == "stats") {
    Serial.println(F("=== STATISTICS ==="));
    Serial.print(F("Predictions: "));
    Serial.println(prediction_count);
    Serial.print(F("Average Brightness: "));
    Serial.println(avg_brightness);
    Serial.print(F("Pot Changes Today: "));
    Serial.println(pot_change_count_today);
    Serial.print(F("Training Samples: "));
    Serial.println(training_sample_count);
    Serial.print(F("Model Type: "));
    Serial.println(use_adaptive_model ? F("ADAPTIVE") : F("RANDOM FOREST"));
  }
  else if (command == "retrain") {
    performRetraining();
  }
  else if (command == "samples") {
    Serial.println(F("=== TRAINING SAMPLES ==="));
    if (training_sample_count == 0) {
      Serial.println(F("No samples collected yet"));
      return;
    }
    Serial.print(F("Total Samples: "));
    Serial.println(training_sample_count);
    
    for (int i = 0; i < training_sample_count; i++) {
      TrainingSample s = loadTrainingSample(i);
      Serial.print(i + 1);
      Serial.print(F(". Ambient="));
      Serial.print(s.ambient_light);
      Serial.print(F(" Motion="));
      Serial.print(s.motion_detected);
      Serial.print(F(" Target="));
      Serial.print(s.target_brightness);
      Serial.print(F("% Time="));
      Serial.println(s.timestamp);
    }
  }
  else if (command == "clearsamples") {
    training_sample_count = 0;
    EEPROM.put(EEPROM_SAMPLE_COUNT_ADDR, training_sample_count);
    EEPROM.commit();
    Serial.println(F("All samples cleared"));
  }
  else if (command == "resetmodel") {
    initializeAdaptiveModel();
    saveAdaptiveModel();
    use_adaptive_model = false;
    Serial.println(F("Model reset to defaults"));
    Serial.println(F("Switched back to RANDOM FOREST model"));
  }
  else if (command == "modelinfo") {
    Serial.println(F("=== MODEL INFO ==="));
    Serial.print(F("Type: "));
    Serial.println(use_adaptive_model ? F("ADAPTIVE (Learned)") : F("RANDOM FOREST (Original)"));
    Serial.print(F("Update Count: "));
    Serial.println(adaptive_model.update_count);
    Serial.print(F("Learning Rate: "));
    Serial.println(adaptive_model.learning_rate, 6);
    Serial.print(F("Bias: "));
    Serial.println(adaptive_model.bias, 2);
  }
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(9600);
  
  while (!Serial) {
    delay(10);
  }
  
  Serial.println(F("==================================="));
  Serial.println(F("LED Controller - Random Forest ML"));
  Serial.println(F("==================================="));
  
  pinMode(PIR_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(LDR_PIN, INPUT);
  pinMode(POT_PIN, INPUT);
  
  // Setup PWM for LED (ESP32 v3.x API)
  ledcAttach(LED_PIN, 5000, 8);
  
  EEPROM.begin(EEPROM_SIZE);
  
  Serial.print(F("Initializing RTC... "));
  
  if (!rtc.begin()) {
    Serial.println(F("FAILED"));
    while (1) {
      delay(1000);
    }
  }
  
  Serial.println(F("OK"));
  
  if (rtc.lostPower()) {
    Serial.println(F("RTC lost power. Setting compile time."));
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }
  
  Serial.print(F("Current Time: "));
  printDateTime(rtc.now());
  Serial.println();
  
  // Initialize training storage
  initializeTrainingStorage();
  
  // Initialize or load adaptive model
  if (training_sample_count > 0) {
    loadAdaptiveModel();
    use_adaptive_model = true;
  } else {
    initializeAdaptiveModel();
  }
  
  smoothed_ldr = readLDR();
  
  Serial.println(F("=== FEATURES ==="));
  Serial.println(F("Random Forest ML prediction"));
  Serial.println(F("Real-time sample collection"));
  Serial.println(F("Gradient descent retraining"));
  Serial.println(F("Persistent model storage"));
  
  Serial.println(F("=== Commands ==="));
  Serial.println(F("help, stats, retrain, samples"));
  
  Serial.println(F("=== Monitoring Started ==="));
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  unsigned long current_time = millis();
  
  if (current_time - last_update >= UPDATE_INTERVAL) {
    last_update = current_time;
    
    DateTime now = rtc.now();
    
    int hour = now.hour();
    int day_of_week = now.dayOfTheWeek();
    int day_of_week_adjusted = (day_of_week + 6) % 7;
    
    float ambient_light = readLDR();
    int motion_detected = digitalRead(PIR_PIN);
    
    smoothed_ldr =
      SMOOTHING_FACTOR * smoothed_ldr +
      (1 - SMOOTHING_FACTOR) * ambient_light;
    
    float sin_hour = sin(2.0 * PI * hour / 24.0);
    float cos_hour = cos(2.0 * PI * hour / 24.0);
    int time_period = getTimePeriod(hour);
    
    // CHOOSE MODEL: Adaptive or Random Forest
    float ml_brightness;
    if (use_adaptive_model) {
      ml_brightness = predictAdaptive(
        smoothed_ldr,
        motion_detected,
        sin_hour,
        cos_hour,
        time_period,
        day_of_week_adjusted
      );
    } else {
      // Use Random Forest prediction from led_rf_model.h
      ml_brightness = predict_led_brightness(
        smoothed_ldr,
        motion_detected,
        hour,
        day_of_week_adjusted
      );
    }
    
    manual_offset = readManualOffset();
    
    float final_brightness = ml_brightness + manual_offset;
    final_brightness = constrain(final_brightness, 0, 100);
    
    smoothed_brightness =
      SMOOTHING_FACTOR * smoothed_brightness +
      (1 - SMOOTHING_FACTOR) * final_brightness;
    
    int pwm_value = map(smoothed_brightness, 0, 100, 0, 255);
    ledcWrite(LED_PIN, pwm_value);
    
    prediction_count++;
    avg_brightness =
      (avg_brightness * (prediction_count - 1) +
       final_brightness) /
      prediction_count;
    
    // Print status
    Serial.print(F("Time: "));
    Serial.print(hour);
    Serial.print(F(":"));
    Serial.print(now.minute());
    Serial.print(F(" | Ambient: "));
    Serial.print(smoothed_ldr);
    Serial.print(F(" | Motion: "));
    Serial.print(motion_detected);
    Serial.print(F(" | ML: "));
    Serial.print(ml_brightness);
    Serial.print(F("% | Offset: "));
    Serial.print(manual_offset);
    Serial.print(F(" | Final: "));
    Serial.print(final_brightness);
    Serial.print(F("% | PWM: "));
    Serial.println(pwm_value);
    
    monitorPotForRetraining(current_time, ml_brightness, final_brightness);
    checkScheduledRetraining(now);
  }
  
  if (Serial.available()) {
    handleSerialCommand();
  }
}
