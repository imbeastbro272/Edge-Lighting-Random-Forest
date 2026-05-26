/*
 * ESP32 Adaptive LED Brightness Control with Random Forest + LIVE RETRAINING
 * 
 * Features:
 * - Random Forest ML predictions with live retraining
 * - Manual/Auto mode switching with potentiometer
 * - Gradient descent adaptive model learning
 * - EEPROM persistent training sample storage
 * - Dual model system: RF base + Adaptive learned
 * - Web interface for monitoring and control
 * - Scheduled retraining (midnight + 20 changes)
 * - Serial command interface for testing
 * 
 * Hardware Requirements:
 * - ESP32 board
 * - LDR (Light Dependent Resistor) for ambient light
 * - PIR motion sensor
 * - Potentiometer for manual brightness adjustment
 * - LED strip (PWM controlled)
 * - RTC module (DS3231)
 * 
 * Pin Configuration:
 * - GPIO34: LDR/Light sensor (ADC)
 * - GPIO35: PIR motion sensor
 * - GPIO33: Potentiometer (ADC) for manual adjustment
 * - GPIO16: LED PWM output
 */

#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>
#include <Wire.h>
#include <RTClib.h>
#include <ArduinoJson.h>
#include "led_rf_model.h"  // Generated model header

// ============================================================================
// CONFIGURATION
// ============================================================================

// WiFi credentials
const char* WIFI_SSID = "YourWiFiSSID";
const char* WIFI_PASSWORD = "YourWiFiPassword";

// Pin definitions
#define PIN_LIGHT_SENSOR 34   // ADC pin for LDR
#define PIN_MOTION_SENSOR 35  // Digital input for PIR
#define PIN_LED_PWM 16        // PWM output for LED
#define PIN_POTENTIOMETER 33  // ADC pin for manual brightness adjustment

// PWM configuration
#define PWM_CHANNEL 0
#define PWM_FREQUENCY 5000
#define PWM_RESOLUTION 8  // 0-255

// Sensor calibration
#define LDR_MIN_VALUE 0
#define LDR_MAX_VALUE 4095
#define LIGHT_MIN_LUX 0
#define LIGHT_MAX_LUX 1000

// Potentiometer configuration
#define POT_CENTER 2048
#define POT_DEADZONE 200
#define POT_STABLE_DURATION 5000  // 5 seconds stable before capturing
#define POT_CHANGE_THRESHOLD 50
#define MAX_CHANGES_BEFORE_RETRAIN 20

// Smoothing
#define SMOOTHING_FACTOR 0.7

// Online learning configuration
#define LEARNING_BUFFER_SIZE 100
#define FEEDBACK_TIMEOUT_MS 30000  // 30 seconds to provide feedback

// Scheduled retraining
#define RETRAIN_HOUR 0
#define RETRAIN_MINUTE 0
#define RETRAIN_SECOND 1

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

RTC_DS3231 rtc;
WebServer server(80);

// Current state
struct SystemState {
  float ambient_light;
  float smoothed_ambient;
  int motion_detected;
  int hour;
  int day_of_week;
  float rf_predicted_brightness;  // Random Forest prediction
  float adaptive_predicted_brightness;  // Adaptive model prediction
  int manual_offset;  // From potentiometer
  float final_brightness;
  float smoothed_final_brightness;
  unsigned long prediction_time;
  bool feedback_pending;
};

SystemState state;

// ============================================================================
// TRAINING SAMPLE STORAGE (EEPROM)
// ============================================================================

#define EEPROM_SIZE 4096
#define MAX_TRAINING_SAMPLES 100
#define SAMPLE_SIZE 32
#define EEPROM_SAMPLES_START 100
#define EEPROM_SAMPLE_COUNT_ADDR 0
#define EEPROM_MODEL_WEIGHTS_START 3300

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
// ADAPTIVE MODEL (Gradient Descent Learning)
// ============================================================================

struct AdaptiveModel {
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

AdaptiveModel adaptive_model;
bool use_adaptive_model = false;  // Start with RF, switch after retraining

// ============================================================================
// POTENTIOMETER MONITORING STATE
// ============================================================================

int last_stable_pot_value = -1;
int current_pot_candidate = -1;
unsigned long pot_stable_start = 0;
bool pot_candidate_active = false;
int pot_change_count_today = 0;

// Retraining flags
bool retrained_at_midnight = false;
bool retrained_at_20_changes = false;
int last_retrain_day = -1;

// Statistics
struct Statistics {
  unsigned long predictions_made;
  unsigned long feedbacks_received;
  float avg_prediction_error;
  unsigned long last_update_time;
};

Statistics stats = {0, 0, 0.0, 0};

// ============================================================================
// ADAPTIVE MODEL FUNCTIONS
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
  
  Serial.println("✓ Adaptive model initialized with defaults");
}

void saveAdaptiveModel() {
  EEPROM.put(EEPROM_MODEL_WEIGHTS_START, adaptive_model);
  EEPROM.commit();
  Serial.println("✓ Model weights saved to EEPROM");
}

void loadAdaptiveModel() {
  EEPROM.get(EEPROM_MODEL_WEIGHTS_START, adaptive_model);
  
  // Validate loaded data
  if (isnan(adaptive_model.bias) || adaptive_model.update_count > 1000000) {
    Serial.println("⚠ Invalid model data, reinitializing");
    initializeAdaptiveModel();
  } else {
    Serial.printf("✓ Loaded adaptive model with %d updates\n", 
                  adaptive_model.update_count);
  }
}

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
  
  Serial.printf("✓ Training samples in EEPROM: %d\n", training_sample_count);
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
  
  Serial.printf("✓ Training sample saved. Total: %d\n", training_sample_count);
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
  sample.ambient_light = state.smoothed_ambient;
  sample.sin_hour = sin(2.0 * PI * now.hour() / 24.0);
  sample.cos_hour = cos(2.0 * PI * now.hour() / 24.0);
  sample.motion_detected = state.motion_detected;
  sample.time_period = getTimePeriod(now.hour());
  sample.day_of_week = (now.dayOfTheWeek() + 6) % 7;  // Convert to Mon=0
  sample.target_brightness = user_brightness;
  sample.timestamp = now.unixtime();
  
  saveTrainingSample(sample);
  
  Serial.printf("📊 Captured: Ambient=%.1f ML=%.1f%% User=%.1f%%\n",
                sample.ambient_light, ml_pred, user_brightness);
}

// ============================================================================
// LIVE RETRAINING (Gradient Descent)
// ============================================================================

void performRetraining() {
  Serial.println("\n╔════════════════════════════════════════════╗");
  Serial.println("║      LIVE RETRAINING IN PROGRESS          ║");
  Serial.println("╚════════════════════════════════════════════╝");
  
  if (training_sample_count < 5) {
    Serial.println("⚠ Insufficient samples (need ≥5). Skipping.");
    return;
  }
  
  Serial.printf("📚 Training on %d samples...\n", training_sample_count);
  
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
  
  Serial.println("\n✓ RETRAINING COMPLETE");
  Serial.printf("  Average Error: %.2f%%\n", total_error);
  Serial.printf("  Total Updates: %d\n", adaptive_model.update_count);
  Serial.println("  Model: ADAPTIVE (User-Learned)");
  
  Serial.println("\n📊 Updated Model Parameters:");
  Serial.printf("  Ambient Weight: %.6f\n", adaptive_model.ambient_weight);
  Serial.printf("  Motion Weight: %.2f\n", adaptive_model.motion_weight);
  Serial.printf("  Bias: %.2f\n", adaptive_model.bias);
  
  Serial.println("╔════════════════════════════════════════════╗");
  Serial.println("║    MODEL ACTIVE - BRIGHTNESS UPDATED       ║");
  Serial.println("╚════════════════════════════════════════════╝\n");
}

// ============================================================================
// POTENTIOMETER MONITORING
// ============================================================================

int readManualOffset() {
  int pot_value = analogRead(PIN_POTENTIOMETER);
  int center = POT_CENTER;
  
  // Deadzone around center (no adjustment)
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

void monitorPotForRetraining(unsigned long current_time) {
  int current_pot_raw = analogRead(PIN_POTENTIOMETER);
  
  if (pot_candidate_active) {
    if (abs(current_pot_raw - current_pot_candidate) <= POT_CHANGE_THRESHOLD) {
      // Pot is stable at candidate value
      if (current_time - pot_stable_start >= POT_STABLE_DURATION) {
        // Pot has been stable long enough
        if (last_stable_pot_value == -1 ||
            abs(current_pot_raw - last_stable_pot_value) > POT_CHANGE_THRESHOLD) {
          
          last_stable_pot_value = current_pot_raw;
          pot_candidate_active = false;
          pot_change_count_today++;
          
          // CAPTURE TRAINING SAMPLE if user made significant adjustment
          if (abs(state.manual_offset) > 5) {
            float ml_pred = use_adaptive_model ? 
                           state.adaptive_predicted_brightness : 
                           state.rf_predicted_brightness;
            captureTrainingSample(ml_pred, state.final_brightness);
          }
          
          Serial.printf("📝 Pot Change #%d\n", pot_change_count_today);
          
          // Trigger retraining at 20 changes
          if (pot_change_count_today >= MAX_CHANGES_BEFORE_RETRAIN &&
              !retrained_at_20_changes) {
            performRetraining();
            retrained_at_20_changes = true;
          }
        }
      }
    } else {
      // Pot moved, reset candidate
      current_pot_candidate = current_pot_raw;
      pot_stable_start = current_time;
    }
  } else {
    // Not monitoring, check for change
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
  
  // Reset counters at midnight
  if (current_day != last_retrain_day && last_retrain_day != -1) {
    pot_change_count_today = 0;
    retrained_at_midnight = false;
    retrained_at_20_changes = false;
    last_retrain_day = current_day;
    Serial.println("🔄 New day - counters reset");
  }
  
  if (last_retrain_day == -1) {
    last_retrain_day = current_day;
  }
  
  // Scheduled retraining at midnight
  if (now.hour() == RETRAIN_HOUR &&
      now.minute() == RETRAIN_MINUTE &&
      now.second() == RETRAIN_SECOND &&
      !retrained_at_midnight) {
    Serial.println("⏰ Scheduled midnight retraining triggered");
    performRetraining();
    retrained_at_midnight = true;
  }
}

int getTimePeriod(int hour) {
  if (hour >= 0 && hour < 6) return 0;      // Early morning
  else if (hour >= 6 && hour < 12) return 1;  // Morning
  else if (hour >= 12 && hour < 17) return 2; // Afternoon
  else if (hour >= 17 && hour < 21) return 3; // Evening
  else return 4;                               // Night
}

// ============================================================================
// TABLE PRINTING
// ============================================================================

void printTableHeader() {
  Serial.println("┌──────────┬─────┬──────┬─────────┬────────┬──────────┬──────────┬────────┬────────┬─────┐");
  Serial.println("│   Time   │ Day │ Hour │ Ambient │ Motion │   RF%    │ Adaptive%│ Offset │ Final% │ PWM │");
  Serial.println("├──────────┼─────┼──────┼─────────┼────────┼──────────┼──────────┼────────┼────────┼─────┤");
}

// ============================================================================
// SETUP
// ============================================================================

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(115200);
  Serial.println("\n\n=================================");
  Serial.println("ESP32 Adaptive Lighting System");
  Serial.println("Random Forest + Live Learning");
  Serial.println("=================================\n");
  
  // Initialize pins
  pinMode(PIN_MOTION_SENSOR, INPUT);
  pinMode(PIN_LIGHT_SENSOR, INPUT);
  pinMode(PIN_POTENTIOMETER, INPUT);
  
  // Setup PWM for LED
  ledcSetup(PWM_CHANNEL, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcAttachPin(PIN_LED_PWM, PWM_CHANNEL);
  
  // Initialize EEPROM
  EEPROM.begin(EEPROM_SIZE);
  Serial.println("✓ EEPROM initialized");
  
  // Initialize training storage
  initializeTrainingStorage();
  
  // Initialize or load adaptive model
  if (training_sample_count > 0) {
    loadAdaptiveModel();
    use_adaptive_model = true;
    Serial.println("✓ Using ADAPTIVE model (learned from user)");
  } else {
    initializeAdaptiveModel();
    Serial.println("✓ Using RANDOM FOREST model (original)");
  }
  
  // Initialize SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("ERROR: SPIFFS mount failed!");
  } else {
    Serial.println("✓ SPIFFS initialized");
  }
  
  // Initialize RTC
  if (!rtc.begin()) {
    Serial.println("WARNING: RTC not found, using compile time");
  } else {
    Serial.println("✓ RTC initialized");
    if (rtc.lostPower()) {
      Serial.println("  RTC lost power, setting time...");
      rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    }
  }
  
  // Connect to WiFi
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int wifi_attempts = 0;
  while (WiFi.status() != WL_CONNECTED && wifi_attempts < 20) {
    delay(500);
    Serial.print(".");
    wifi_attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓ WiFi connected");
    Serial.print("  IP address: ");
    Serial.println(WiFi.localIP());
    
    // Setup web server
    setupWebServer();
    server.begin();
    Serial.println("✓ Web server started");
  } else {
    Serial.println("\n⚠ WiFi connection failed, running offline");
  }
  
  // Initialize state
  state.feedback_pending = false;
  state.prediction_time = 0;
  state.smoothed_ambient = readLDR();
  state.smoothed_final_brightness = 0;
  
  Serial.println("\n=== LIVE LEARNING FEATURES ===");
  Serial.println("✓ Dual model: Random Forest + Adaptive");
  Serial.println("✓ Potentiometer manual override");
  Serial.println("✓ Real-time sample collection");
  Serial.println("✓ Gradient descent retraining");
  Serial.println("✓ Persistent EEPROM storage");
  Serial.println("✓ Web interface + Serial commands");
  
  Serial.println("\n=== Serial Commands ===");
  Serial.println("help, stats, retrain, samples, clearsamples");
  Serial.println("test HH DD AMBIENT MOTION POT");
  Serial.println("weights, modelinfo, resetmodel");
  
  Serial.println("\n=== Monitoring Started ===");
  printTableHeader();
  
  // Initial reading
  updateSensors();
  makePrediction();
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  unsigned long current_time = millis();
  
  // Handle web server requests
  server.handleClient();
  
  // Update sensors every second
  static unsigned long lastSensorUpdate = 0;
  if (current_time - lastSensorUpdate >= 1000) {
    updateSensors();
    lastSensorUpdate = current_time;
  }
  
  // Make prediction every 10 seconds or when motion changes
  static unsigned long lastPrediction = 0;
  static int lastMotion = 0;
  
  if (state.motion_detected != lastMotion || 
      current_time - lastPrediction >= 10000) {
    makePrediction();
    lastPrediction = current_time;
    lastMotion = state.motion_detected;
  }
  
  // Monitor potentiometer for user adjustments
  monitorPotForRetraining(current_time);
  
  // Check scheduled retraining
  DateTime now = rtc.now();
  checkScheduledRetraining(now);
  
  // Check for feedback timeout
  if (state.feedback_pending && 
      current_time - state.prediction_time >= FEEDBACK_TIMEOUT_MS) {
    // No feedback received, assume prediction was good
    Serial.println("No feedback timeout - accepting prediction");
    acceptPrediction();
  }
  
  // Periodic status report
  static unsigned long lastStatus = 0;
  if (current_time - lastStatus >= 60000) {  // Every minute
    printStatus();
    lastStatus = current_time;
  }
  
  // Handle serial commands
  if (Serial.available()) {
    handleSerialCommand();
  }
  
  delay(10);
}

// ============================================================================
// SENSOR FUNCTIONS
// ============================================================================

float readLDR() {
  long sum = 0;
  for (int i = 0; i < 20; i++) {
    sum += analogRead(PIN_LIGHT_SENSOR);
    delay(2);
  }
  
  float avg = sum / 20.0;
  float voltage = (avg / 4095.0) * 3.3;
  
  if (voltage <= 0.01) return 0;
  
  // Convert to lux (approximate formula for LDR)
  float r_ldr = 10000.0 * voltage / (3.3 - voltage);
  float lux = 32768000.0 * pow(r_ldr, -1.4);
  lux = constrain(lux, 0, 100000);
  
  return lux;
}

void updateSensors() {
  // Read light sensor
  float ldr_value = readLDR();
  state.ambient_light = ldr_value;
  
  // Smooth ambient light
  state.smoothed_ambient = SMOOTHING_FACTOR * state.smoothed_ambient +
                           (1 - SMOOTHING_FACTOR) * ldr_value;
  
  // Read motion sensor
  state.motion_detected = digitalRead(PIN_MOTION_SENSOR);
  
  // Get current time
  DateTime now = rtc.now();
  state.hour = now.hour();
  state.day_of_week = (now.dayOfTheWeek() + 6) % 7;  // Convert to Mon=0
}

// ============================================================================
// PREDICTION FUNCTIONS
// ============================================================================

void makePrediction() {
  DateTime now = rtc.now();
  
  float sin_hour = sin(2.0 * PI * state.hour / 24.0);
  float cos_hour = cos(2.0 * PI * state.hour / 24.0);
  int time_period = getTimePeriod(state.hour);
  
  // Get Random Forest prediction (using compiled C++ model)
  state.rf_predicted_brightness = predict_led_brightness(
    state.smoothed_ambient,
    state.motion_detected,
    state.hour,
    state.day_of_week
  );
  
  // Get Adaptive model prediction
  state.adaptive_predicted_brightness = predictAdaptive(
    state.smoothed_ambient,
    state.motion_detected,
    sin_hour,
    cos_hour,
    time_period,
    state.day_of_week
  );
  
  // Choose which model to use
  float ml_brightness = use_adaptive_model ? 
                        state.adaptive_predicted_brightness : 
                        state.rf_predicted_brightness;
  
  // Read manual offset from potentiometer
  state.manual_offset = readManualOffset();
  
  // Calculate final brightness
  state.final_brightness = ml_brightness + state.manual_offset;
  state.final_brightness = constrain(state.final_brightness, 0, 100);
  
  // Smooth final brightness
  state.smoothed_final_brightness = SMOOTHING_FACTOR * state.smoothed_final_brightness +
                                     (1 - SMOOTHING_FACTOR) * state.final_brightness;
  
  // Apply brightness
  setBrightness((int)state.smoothed_final_brightness);
  
  state.prediction_time = millis();
  state.feedback_pending = true;
  
  // Update stats
  stats.predictions_made++;
  
  // Print table row
  printPredictionRow(now, ml_brightness);
}

void setBrightness(int brightness_percent) {
  // Convert percentage to PWM value (0-255)
  int pwm_value = map(brightness_percent, 0, 100, 0, 255);
  ledcWrite(PWM_CHANNEL, pwm_value);
}

void printPredictionRow(DateTime now, float ml_pred) {
  char time_str[9];
  sprintf(time_str, "%02d:%02d:%02d", now.hour(), now.minute(), now.second());
  
  const char *day_names[] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
  
  int pwm_value = map((int)state.smoothed_final_brightness, 0, 100, 0, 255);
  
  char line[160];
  sprintf(line,
    "│ %s │ %3s │  %2d  │ %6.1f  │   %d    │  %6.1f  │  %6.1f  │ %+5d  │ %5.1f  │ %3d │",
    time_str,
    day_names[state.day_of_week],
    state.hour,
    state.smoothed_ambient,
    state.motion_detected,
    state.rf_predicted_brightness,
    state.adaptive_predicted_brightness,
    state.manual_offset,
    state.final_brightness,
    pwm_value
  );
  
  Serial.println(line);
}

void acceptPrediction() {
  if (!state.feedback_pending) return;
  
  // Add to training samples
  float ml_pred = use_adaptive_model ? 
                  state.adaptive_predicted_brightness : 
                  state.rf_predicted_brightness;
  captureTrainingSample(ml_pred, state.final_brightness);
  
  state.feedback_pending = false;
  stats.feedbacks_received++;
  
  Serial.println("✓ Prediction accepted and added to learning buffer");
}

void rejectPrediction(int user_brightness) {
  if (!state.feedback_pending) return;
  
  // User provided different brightness
  setBrightness(user_brightness);
  state.final_brightness = user_brightness;
  
  // Add corrected data to learning buffer
  float ml_pred = use_adaptive_model ? 
                  state.adaptive_predicted_brightness : 
                  state.rf_predicted_brightness;
  captureTrainingSample(ml_pred, user_brightness);
  
  // Update error statistics
  int error = abs((int)ml_pred - user_brightness);
  stats.avg_prediction_error = (stats.avg_prediction_error * stats.feedbacks_received + error) 
                                / (stats.feedbacks_received + 1);
  
  state.feedback_pending = false;
  stats.feedbacks_received++;
  
  Serial.printf("✓ User feedback: %d%% (error: %d%%)\n", 
                user_brightness, error);
}

// ============================================================================
// SERIAL COMMAND HANDLING
// ============================================================================

void handleSerialCommand() {
  String command = Serial.readStringUntil('\n');
  command.trim();
  
  if (command == "help") {
    Serial.println("\n=== BASIC COMMANDS ===");
    Serial.println("help          - Show all commands");
    Serial.println("stats         - Show statistics");
    Serial.println("retrain       - Force retraining");
    Serial.println("");
    Serial.println("=== SAMPLE MANAGEMENT ===");
    Serial.println("samples       - View training samples");
    Serial.println("clearsamples  - Clear all samples");
    Serial.println("");
    Serial.println("=== MODEL INSPECTION ===");
    Serial.println("modelinfo     - Show model summary");
    Serial.println("weights       - Show detailed weights");
    Serial.println("resetmodel    - Reset adaptive model");
    Serial.println("");
    Serial.println("=== MANUAL TESTING ===");
    Serial.println("test HH DD AMBIENT MOTION POT");
    Serial.println("  HH: Hour (0-23)");
    Serial.println("  DD: Day (0=Mon, 1=Tue, ..., 6=Sun)");
    Serial.println("  AMBIENT: Light level in lux (0-10000)");
    Serial.println("  MOTION: 0=No, 1=Yes");
    Serial.println("  POT: Pot value (0-4095, 2048=center)");
    Serial.println("");
    Serial.println("Example: test 22 1 300 1 2200");
    Serial.println("         (10 PM, Tuesday, 300 lux, motion, pot=2200)");
  }
  else if (command == "stats") {
    Serial.println("\n=== STATISTICS ===");
    Serial.printf("Predictions: %lu\n", stats.predictions_made);
    Serial.printf("Feedbacks Received: %lu\n", stats.feedbacks_received);
    Serial.printf("Avg Prediction Error: %.2f%%\n", stats.avg_prediction_error);
    Serial.printf("Pot Changes Today: %d\n", pot_change_count_today);
    Serial.printf("Training Samples: %d\n", training_sample_count);
    Serial.printf("Model Type: %s\n", use_adaptive_model ? "ADAPTIVE" : "RANDOM FOREST");
    Serial.printf("Uptime: %lu seconds\n", millis() / 1000);
  }
  else if (command == "retrain") {
    performRetraining();
  }
  else if (command == "samples") {
    Serial.println("\n=== TRAINING SAMPLES ===");
    if (training_sample_count == 0) {
      Serial.println("No samples collected yet");
      return;
    }
    Serial.printf("Total Samples: %d\n\n", training_sample_count);
    Serial.println("┌────┬─────────┬────────┬──────────┬─────────┬──────────┬─────────────┐");
    Serial.println("│ #  │Ambient  │Motion  │ Period   │ Day     │ Target%  │  Timestamp  │");
    Serial.println("├────┼─────────┼────────┼──────────┼─────────┼──────────┼─────────────┤");
    
    for (int i = 0; i < training_sample_count; i++) {
      TrainingSample s = loadTrainingSample(i);
      const char* day_names[] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
      const char* period_names[] = {"E.Morn", "Morn", "Aftern", "Even", "Night"};
      
      char line[100];
      sprintf(line, "│%3d │%7.1f │   %d    │ %-8s │ %-7s │  %6.1f  │ %lu │",
              i + 1, s.ambient_light, s.motion_detected, 
              period_names[s.time_period], day_names[s.day_of_week],
              s.target_brightness, s.timestamp);
      Serial.println(line);
    }
    Serial.println("└────┴─────────┴────────┴──────────┴─────────┴──────────┴─────────────┘");
  }
  else if (command == "clearsamples") {
    training_sample_count = 0;
    EEPROM.put(EEPROM_SAMPLE_COUNT_ADDR, training_sample_count);
    EEPROM.commit();
    Serial.println("✓ All samples cleared");
  }
  else if (command == "resetmodel") {
    initializeAdaptiveModel();
    saveAdaptiveModel();
    use_adaptive_model = false;
    Serial.println("✓ Model reset to defaults");
    Serial.println("  Switched back to RANDOM FOREST model");
  }
  else if (command == "modelinfo") {
    Serial.println("\n=== MODEL INFO ===");
    Serial.printf("Active Type: %s\n", use_adaptive_model ? "ADAPTIVE (Learned)" : "RANDOM FOREST (Original)");
    Serial.printf("Update Count: %d\n", adaptive_model.update_count);
    Serial.printf("Learning Rate: %.6f\n", adaptive_model.learning_rate);
    Serial.printf("Bias: %.2f\n", adaptive_model.bias);
    Serial.printf("Training Samples: %d/%d\n", training_sample_count, MAX_TRAINING_SAMPLES);
  }
  else if (command == "weights") {
    displayModelWeights();
  }
  else if (command.startsWith("test ")) {
    int hour, day, motion, pot;
    float ambient;
    
    int parsed = sscanf(command.c_str(), "test %d %d %f %d %d",
                       &hour, &day, &ambient, &motion, &pot);
    
    if (parsed == 5) {
      if (hour < 0 || hour > 23) {
        Serial.println("❌ Error: Hour must be 0-23");
        return;
      }
      if (day < 0 || day > 6) {
        Serial.println("❌ Error: Day must be 0-6 (0=Mon, 6=Sun)");
        return;
      }
      if (ambient < 0 || ambient > 100000) {
        Serial.println("❌ Error: Ambient must be 0-100000 lux");
        return;
      }
      if (motion != 0 && motion != 1) {
        Serial.println("❌ Error: Motion must be 0 or 1");
        return;
      }
      if (pot < 0 || pot > 4095) {
        Serial.println("❌ Error: Pot must be 0-4095");
        return;
      }
      
      testManualPrediction(hour, day, ambient, motion, pot);
    } else {
      Serial.println("❌ Invalid format. Use: test HH DD AMBIENT MOTION POT");
      Serial.println("   Example: test 22 1 300 1 2200");
    }
  }
  else {
    Serial.printf("Unknown command: %s\n", command.c_str());
    Serial.println("Type 'help' for available commands");
  }
}

void testManualPrediction(int hour, int day, float ambient, int motion, int pot_value) {
  float sin_hour = sin(2.0 * PI * hour / 24.0);
  float cos_hour = cos(2.0 * PI * hour / 24.0);
  int time_period = getTimePeriod(hour);
  
  // Calculate offset from pot value
  int center = POT_CENTER;
  int offset = 0;
  
  if (pot_value >= center - POT_DEADZONE &&
      pot_value <= center + POT_DEADZONE) {
    offset = 0;
  }
  else if (pot_value < center - POT_DEADZONE) {
    offset = map(pot_value, 0, center - POT_DEADZONE, -100, 0);
  }
  else {
    offset = map(pot_value, center + POT_DEADZONE, 4095, 0, 100);
  }
  
  // Get predictions from both models
  float rf_pred = predict_led_brightness(ambient, motion, hour, day);
  
  float adaptive_pred = predictAdaptive(
    ambient, motion, sin_hour, cos_hour, time_period, day
  );
  
  // Calculate final brightness
  float rf_final = constrain(rf_pred + offset, 0, 100);
  float adaptive_final = constrain(adaptive_pred + offset, 0, 100);
  
  // Display results
  Serial.println("\n╔════════════════════════════════════════════╗");
  Serial.println("║     MANUAL PREDICTION TEST RESULTS         ║");
  Serial.println("╚════════════════════════════════════════════╝");
  
  const char* day_names[] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
  const char* period_names[] = {"Early Morning", "Morning", "Afternoon", "Evening", "Night"};
  
  Serial.println("\n📍 Input Parameters:");
  Serial.printf("   Time: %d:00 (%s)\n", hour, period_names[time_period]);
  Serial.printf("   Day: %s\n", day_names[day]);
  Serial.printf("   Ambient Light: %.1f lux\n", ambient);
  Serial.printf("   Motion: %s\n", motion ? "YES" : "NO");
  Serial.printf("   Pot Value: %d (Offset: %+d%%)\n", pot_value, offset);
  
  Serial.println("\n📊 Model Predictions:");
  Serial.println("┌─────────────────────┬──────────┬──────────┬──────────┐");
  Serial.println("│ Model               │ Raw (%)  │ Offset   │ Final(%) │");
  Serial.println("├─────────────────────┼──────────┼──────────┼──────────┤");
  
  char rf_line[70];
  sprintf(rf_line, "│ RANDOM FOREST (Base)│ %7.2f │ %+7d │ %7.2f │",
          rf_pred, offset, rf_final);
  Serial.println(rf_line);
  
  char adaptive_line[70];
  sprintf(adaptive_line, "│ ADAPTIVE (Learned)  │ %7.2f │ %+7d │ %7.2f │",
          adaptive_pred, offset, adaptive_final);
  Serial.println(adaptive_line);
  
  Serial.println("└─────────────────────┴──────────┴──────────┴──────────┘");
  
  float diff = adaptive_pred - rf_pred;
  float diff_pct = (rf_pred > 0) ? (diff / rf_pred) * 100 : 0;
  
  Serial.println("\n📈 Model Comparison:");
  Serial.printf("   Difference: %.2f%% (%.1f%% change)\n", diff, diff_pct);
  Serial.printf("   Active Model: %s\n", 
                use_adaptive_model ? "ADAPTIVE (Learned)" : "RANDOM FOREST (Original)");
  
  if (use_adaptive_model) {
    Serial.println("   ✓ Using learned model for real control");
  } else {
    Serial.println("   ✓ Using original RF (adaptive available after retraining)");
  }
  
  Serial.println();
}

void displayModelWeights() {
  Serial.println("\n╔════════════════════════════════════════════╗");
  Serial.println("║    DETAILED MODEL WEIGHTS (LEARNED)        ║");
  Serial.println("╚════════════════════════════════════════════╝");
  
  Serial.printf("\nBias: %.3f\n", adaptive_model.bias);
  
  Serial.println("\nFeature Weights:");
  Serial.printf("  Ambient Light: %.6f\n", adaptive_model.ambient_weight);
  Serial.printf("  Motion: %.3f\n", adaptive_model.motion_weight);
  Serial.printf("  Sin(Hour): %.3f\n", adaptive_model.sin_hour_weight);
  Serial.printf("  Cos(Hour): %.3f\n", adaptive_model.cos_hour_weight);
  
  Serial.println("\nTime Period Weights:");
  const char* periods[] = {"Early Morning", "Morning", "Afternoon", "Evening", "Night"};
  for (int i = 0; i < 5; i++) {
    Serial.printf("  %s: %.3f\n", periods[i], adaptive_model.time_period_weights[i]);
  }
  
  Serial.println("\nDay of Week Weights:");
  const char* days[] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
  for (int i = 0; i < 7; i++) {
    Serial.printf("  %s: %.3f\n", days[i], adaptive_model.day_of_week_weights[i]);
  }
  
  Serial.printf("\nLearning Rate: %.6f\n", adaptive_model.learning_rate);
  Serial.printf("Total Updates: %d\n", adaptive_model.update_count);
  Serial.println();
}

// ============================================================================
// WEB SERVER FUNCTIONS
// ============================================================================

void setupWebServer() {
  // Main page
  server.on("/", HTTP_GET, handleRoot);
  
  // API endpoints
  server.on("/api/status", HTTP_GET, handleStatus);
  server.on("/api/feedback", HTTP_POST, handleFeedback);
  server.on("/api/manual", HTTP_POST, handleManual);
  server.on("/api/retrain", HTTP_POST, handleRetrain);
  server.on("/api/export", HTTP_GET, handleExport);
  server.on("/api/stats", HTTP_GET, handleStats);
  
  Serial.println("✓ Web server routes configured");
}

void handleRoot() {
  String html = R"(
<!DOCTYPE html>
<html>
<head>
  <title>ESP32 Adaptive Lighting - RF + Learning</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial; margin: 20px; background: #f0f0f0; }
    .container { max-width: 700px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #333; text-align: center; margin-bottom: 5px; }
    h4 { text-align: center; color: #666; margin-top: 0; }
    .status { padding: 15px; margin: 10px 0; background: #e8f4f8; border-radius: 5px; }
    .status-item { margin: 8px 0; }
    .label { font-weight: bold; color: #555; }
    .value { color: #007bff; }
    .model-info { display: flex; justify-content: space-between; margin: 10px 0; }
    .model-box { flex: 1; padding: 10px; margin: 0 5px; background: #f9f9f9; border-radius: 5px; border: 2px solid #ddd; }
    .model-box.active { border-color: #28a745; background: #e8f5e9; }
    .model-box h5 { margin: 0 0 8px 0; color: #333; }
    button { padding: 12px 24px; margin: 5px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; }
    .btn-accept { background: #28a745; color: white; }
    .btn-reject { background: #dc3545; color: white; }
    .btn-manual { background: #007bff; color: white; }
    .btn-retrain { background: #ff9800; color: white; }
    .slider { width: 100%; margin: 10px 0; }
    #brightness-value { font-size: 24px; color: #007bff; font-weight: bold; }
    .info-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🔆 Adaptive Lighting Control</h1>
    <h4>Random Forest + Live Learning</h4>
    
    <div class="status">
      <h2>Current Status</h2>
      <div class="status-item"><span class="label">Ambient Light:</span> <span class="value" id="ambient">--</span> lux</div>
      <div class="status-item"><span class="label">Motion:</span> <span class="value" id="motion">--</span></div>
      <div class="status-item"><span class="label">Time:</span> <span class="value" id="time">--</span></div>
      <div class="status-item"><span class="label">Manual Offset:</span> <span class="value" id="offset">--</span>%</div>
      <div class="status-item"><span class="label">Final Brightness:</span> <span class="value" id="final">--</span>%</div>
    </div>
    
    <div class="status">
      <h2>Model Predictions</h2>
      <div class="model-info">
        <div class="model-box" id="rf-box">
          <h5>🌲 Random Forest</h5>
          <div><span class="label">Predicted:</span> <span class="value" id="rf-pred">--</span>%</div>
        </div>
        <div class="model-box" id="adaptive-box">
          <h5>🧠 Adaptive (Learned)</h5>
          <div><span class="label">Predicted:</span> <span class="value" id="adaptive-pred">--</span>%</div>
        </div>
      </div>
      <div style="text-align: center; margin-top: 10px;">
        <strong>Active Model:</strong> <span class="value" id="active-model">--</span>
      </div>
    </div>
    
    <div class="status">
      <h2>Manual Control</h2>
      <input type="range" min="0" max="100" value="50" class="slider" id="brightness-slider" oninput="updateSlider(this.value)">
      <p><span id="brightness-value">50</span>%</p>
      <button class="btn-manual" onclick="sendManual()">Set Brightness</button>
    </div>
    
    <div class="status">
      <h2>Statistics</h2>
      <div class="info-grid">
        <div><span class="label">Predictions:</span> <span class="value" id="predictions">--</span></div>
        <div><span class="label">Feedbacks:</span> <span class="value" id="feedbacks">--</span></div>
        <div><span class="label">Training Samples:</span> <span class="value" id="samples">--</span></div>
        <div><span class="label">Pot Changes Today:</span> <span class="value" id="pot-changes">--</span></div>
        <div><span class="label">Model Updates:</span> <span class="value" id="model-updates">--</span></div>
        <div><span class="label">Avg Error:</span> <span class="value" id="error">--</span>%</div>
      </div>
    </div>
    
    <div style="text-align: center; margin-top: 20px;">
      <button class="btn-retrain" onclick="performRetrain()">🔄 Retrain Model Now</button>
      <button class="btn-manual" onclick="exportData()">📥 Export Training Data</button>
    </div>
  </div>
  
  <script>
    function updateStatus() {
      fetch('/api/status')
        .then(r => r.json())
        .then(data => {
          document.getElementById('ambient').textContent = data.smoothed_ambient.toFixed(0);
          document.getElementById('motion').textContent = data.motion_detected ? 'Detected' : 'None';
          document.getElementById('time').textContent = data.hour + ':00, Day ' + data.day_of_week;
          document.getElementById('offset').textContent = data.manual_offset >= 0 ? '+' + data.manual_offset : data.manual_offset;
          document.getElementById('final').textContent = data.actual_brightness;
          
          document.getElementById('rf-pred').textContent = data.rf_predicted.toFixed(1);
          document.getElementById('adaptive-pred').textContent = data.adaptive_predicted.toFixed(1);
          
          let activeModel = data.use_adaptive_model ? 'ADAPTIVE (Learned)' : 'RANDOM FOREST (Original)';
          document.getElementById('active-model').textContent = activeModel;
          
          // Highlight active model
          if (data.use_adaptive_model) {
            document.getElementById('adaptive-box').classList.add('active');
            document.getElementById('rf-box').classList.remove('active');
          } else {
            document.getElementById('rf-box').classList.add('active');
            document.getElementById('adaptive-box').classList.remove('active');
          }
        });
      
      fetch('/api/stats')
        .then(r => r.json())
        .then(data => {
          document.getElementById('predictions').textContent = data.predictions_made;
          document.getElementById('feedbacks').textContent = data.feedbacks_received;
          document.getElementById('samples').textContent = data.training_sample_count + '/' + data.buffer_size;
          document.getElementById('pot-changes').textContent = data.pot_changes_today;
          document.getElementById('model-updates').textContent = data.model_updates;
          document.getElementById('error').textContent = data.avg_error.toFixed(1);
        });
    }
    
    function sendManual() {
      let brightness = document.getElementById('brightness-slider').value;
      fetch('/api/manual', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({brightness: parseInt(brightness)})
      }).then(() => {
        alert('Brightness set to ' + brightness + '%');
        updateStatus();
      });
    }
    
    function performRetrain() {
      if (!confirm('Retrain the adaptive model now with collected samples?')) return;
      
      fetch('/api/retrain', {
        method: 'POST'
      }).then(r => r.text()).then(msg => {
        alert(msg);
        updateStatus();
      });
    }
    
    function updateSlider(val) {
      document.getElementById('brightness-value').textContent = val;
    }
    
    function exportData() {
      window.location.href = '/api/export';
    }
    
    setInterval(updateStatus, 2000);
    updateStatus();
  </script>
</body>
</html>
  )";
  
  server.send(200, "text/html", html);
}

void handleStatus() {
  StaticJsonDocument<384> doc;
  doc["ambient_light"] = state.ambient_light;
  doc["smoothed_ambient"] = state.smoothed_ambient;
  doc["motion_detected"] = state.motion_detected;
  doc["hour"] = state.hour;
  doc["day_of_week"] = state.day_of_week;
  doc["rf_predicted"] = state.rf_predicted_brightness;
  doc["adaptive_predicted"] = state.adaptive_predicted_brightness;
  doc["manual_offset"] = state.manual_offset;
  doc["final_brightness"] = state.final_brightness;
  doc["actual_brightness"] = (int)state.smoothed_final_brightness;
  doc["feedback_pending"] = state.feedback_pending;
  doc["use_adaptive_model"] = use_adaptive_model;
  doc["pot_changes_today"] = pot_change_count_today;
  
  String json;
  serializeJson(doc, json);
  server.send(200, "application/json", json);
}

void handleStats() {
  StaticJsonDocument<384> doc;
  doc["predictions_made"] = stats.predictions_made;
  doc["feedbacks_received"] = stats.feedbacks_received;
  doc["avg_error"] = stats.avg_prediction_error;
  doc["training_sample_count"] = training_sample_count;
  doc["buffer_size"] = MAX_TRAINING_SAMPLES;
  doc["pot_changes_today"] = pot_change_count_today;
  doc["model_type"] = use_adaptive_model ? "ADAPTIVE" : "RANDOM_FOREST";
  doc["model_updates"] = adaptive_model.update_count;
  doc["uptime"] = millis() / 1000;
  
  String json;
  serializeJson(doc, json);
  server.send(200, "application/json", json);
}

void handleExport() {
  // Export training data as CSV
  String csv = "ambient_light,motion_detected,sin_hour,cos_hour,time_period,day_of_week,led_brightness,timestamp\n";
  
  for (int i = 0; i < training_sample_count; i++) {
    TrainingSample sample = loadTrainingSample(i);
    csv += String(sample.ambient_light, 2) + ",";
    csv += String(sample.motion_detected) + ",";
    csv += String(sample.sin_hour, 4) + ",";
    csv += String(sample.cos_hour, 4) + ",";
    csv += String(sample.time_period) + ",";
    csv += String(sample.day_of_week) + ",";
    csv += String(sample.target_brightness, 1) + ",";
    csv += String(sample.timestamp) + "\n";
  }
  
  server.send(200, "text/csv", csv);
  Serial.println("✓ Training data exported via web interface");
}

void handleFeedback() {
  StaticJsonDocument<128> doc;
  deserializeJson(doc, server.arg("plain"));
  
  String action = doc["action"];
  
  if (action == "accept") {
    acceptPrediction();
    server.send(200, "text/plain", "Feedback accepted");
  } else if (action == "reject") {
    int brightness = doc["brightness"];
    rejectPrediction(brightness);
    server.send(200, "text/plain", "Feedback with correction applied");
  }
}

void handleManual() {
  StaticJsonDocument<64> doc;
  deserializeJson(doc, server.arg("plain"));
  
  int brightness = doc["brightness"];
  setBrightness(brightness);
  state.final_brightness = brightness;
  
  // Capture training sample
  float ml_pred = use_adaptive_model ? 
                  state.adaptive_predicted_brightness : 
                  state.rf_predicted_brightness;
  captureTrainingSample(ml_pred, brightness);
  
  server.send(200, "text/plain", "Manual brightness set");
}

void handleRetrain() {
  performRetraining();
  server.send(200, "text/plain", "Retraining completed");
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void printStatus() {
  Serial.println("\n========== STATUS REPORT ==========");
  Serial.printf("Uptime: %lu seconds\n", millis() / 1000);
  Serial.printf("Predictions Made: %lu\n", stats.predictions_made);
  Serial.printf("Feedbacks Received: %lu\n", stats.feedbacks_received);
  Serial.printf("Learning Buffer: %d/%d\n", bufferCount, LEARNING_BUFFER_SIZE);
  Serial.printf("Average Error: %.2f%%\n", stats.avg_prediction_error);
  Serial.printf("WiFi Status: %s\n", WiFi.status() == WL_CONNECTED ? "Connected" : "Disconnected");
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("IP Address: %s\n", WiFi.localIP().toString().c_str());
  }
  Serial.println("==================================\n");
}
