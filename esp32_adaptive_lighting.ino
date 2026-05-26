/*
 * ESP32 Adaptive LED Brightness Control with Random Forest
 * 
 * Features:
 * - Live sensor input (ambient light, motion)
 * - ML-based brightness prediction
 * - User feedback collection for online learning
 * - WiFi connectivity for model updates
 * - SPIFFS storage for learning data
 * - Web interface for monitoring and control
 * 
 * Hardware Requirements:
 * - ESP32 board
 * - LDR (Light Dependent Resistor) or BH1750 light sensor
 * - PIR motion sensor
 * - LED strip (PWM controlled)
 * - RTC module (DS3231 or use NTP)
 * 
 * Pin Configuration:
 * - GPIO34: LDR/Light sensor (ADC)
 * - GPIO35: PIR motion sensor
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

// PWM configuration
#define PWM_CHANNEL 0
#define PWM_FREQUENCY 5000
#define PWM_RESOLUTION 8  // 0-255

// Sensor calibration
#define LDR_MIN_VALUE 0
#define LDR_MAX_VALUE 4095
#define LIGHT_MIN_LUX 0
#define LIGHT_MAX_LUX 1000

// Online learning configuration
#define LEARNING_BUFFER_SIZE 100
#define FEEDBACK_TIMEOUT_MS 30000  // 30 seconds to provide feedback

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

RTC_DS3231 rtc;
WebServer server(80);

// Current state
struct SystemState {
  float ambient_light;
  int motion_detected;
  int hour;
  int day_of_week;
  int predicted_brightness;
  int actual_brightness;
  unsigned long prediction_time;
  bool feedback_pending;
};

SystemState state;

// Online learning buffer
struct LearningData {
  float ambient_light;
  int motion_detected;
  float sin_hour;
  float cos_hour;
  int time_period;
  int day_of_week;
  int brightness;
  unsigned long timestamp;
};

LearningData learningBuffer[LEARNING_BUFFER_SIZE];
int bufferIndex = 0;
int bufferCount = 0;

// Statistics
struct Statistics {
  unsigned long predictions_made;
  unsigned long feedbacks_received;
  float avg_prediction_error;
  unsigned long last_update_time;
};

Statistics stats = {0, 0, 0.0, 0};

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(115200);
  Serial.println("\n\n=================================");
  Serial.println("ESP32 Adaptive Lighting System");
  Serial.println("Random Forest ML Model");
  Serial.println("=================================\n");
  
  // Initialize pins
  pinMode(PIN_MOTION_SENSOR, INPUT);
  pinMode(PIN_LIGHT_SENSOR, INPUT);
  
  // Setup PWM for LED
  ledcSetup(PWM_CHANNEL, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcAttachPin(PIN_LED_PWM, PWM_CHANNEL);
  
  // Initialize SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("ERROR: SPIFFS mount failed!");
  } else {
    Serial.println("✓ SPIFFS initialized");
    loadLearningBuffer();
  }
  
  // Initialize RTC
  if (!rtc.begin()) {
    Serial.println("WARNING: RTC not found, using compile time");
    // Use compile time as fallback
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
  
  Serial.println("\n=================================");
  Serial.println("System Ready!");
  Serial.println("=================================\n");
  
  // Initial brightness reading
  updateSensors();
  makePrediction();
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  // Handle web server requests
  server.handleClient();
  
  // Update sensors every second
  static unsigned long lastSensorUpdate = 0;
  if (millis() - lastSensorUpdate >= 1000) {
    updateSensors();
    lastSensorUpdate = millis();
  }
  
  // Make prediction every 10 seconds or when motion detected
  static unsigned long lastPrediction = 0;
  static int lastMotion = 0;
  
  if (state.motion_detected != lastMotion || 
      millis() - lastPrediction >= 10000) {
    makePrediction();
    lastPrediction = millis();
    lastMotion = state.motion_detected;
  }
  
  // Check for feedback timeout
  if (state.feedback_pending && 
      millis() - state.prediction_time >= FEEDBACK_TIMEOUT_MS) {
    // No feedback received, assume prediction was good
    Serial.println("No feedback timeout - accepting prediction");
    acceptPrediction();
  }
  
  // Periodic status report
  static unsigned long lastStatus = 0;
  if (millis() - lastStatus >= 60000) {  // Every minute
    printStatus();
    lastStatus = millis();
  }
  
  delay(10);
}

// ============================================================================
// SENSOR FUNCTIONS
// ============================================================================

void updateSensors() {
  // Read light sensor (LDR)
  int ldr_value = analogRead(PIN_LIGHT_SENSOR);
  state.ambient_light = map(ldr_value, LDR_MIN_VALUE, LDR_MAX_VALUE, 
                            LIGHT_MIN_LUX, LIGHT_MAX_LUX);
  
  // Read motion sensor
  state.motion_detected = digitalRead(PIN_MOTION_SENSOR);
  
  // Get current time
  DateTime now = rtc.now();
  state.hour = now.hour();
  state.day_of_week = get_day_of_week_from_rtc(now.dayOfTheWeek());
}

// ============================================================================
// PREDICTION FUNCTIONS
// ============================================================================

void makePrediction() {
  // Get ML prediction
  int predicted = predict_led_brightness(
    state.ambient_light,
    state.motion_detected,
    state.hour,
    state.day_of_week
  );
  
  state.predicted_brightness = predicted;
  state.prediction_time = millis();
  state.feedback_pending = true;
  
  // Apply brightness
  setBrightness(predicted);
  state.actual_brightness = predicted;
  
  // Log prediction
  stats.predictions_made++;
  
  Serial.println("\n--- New Prediction ---");
  Serial.printf("Time: %02d:00, Day: %d\n", state.hour, state.day_of_week);
  Serial.printf("Ambient: %.0f lux, Motion: %d\n", 
                state.ambient_light, state.motion_detected);
  Serial.printf("Predicted Brightness: %d%%\n", predicted);
  Serial.println("--------------------");
}

void setBrightness(int brightness_percent) {
  // Convert percentage to PWM value (0-255)
  int pwm_value = map(brightness_percent, 0, 100, 0, 255);
  ledcWrite(PWM_CHANNEL, pwm_value);
  
  Serial.printf("LED Brightness set to: %d%% (PWM: %d)\n", 
                brightness_percent, pwm_value);
}

void acceptPrediction() {
  if (!state.feedback_pending) return;
  
  // Add to learning buffer
  addToLearningBuffer(
    state.ambient_light,
    state.motion_detected,
    state.hour,
    state.day_of_week,
    state.actual_brightness
  );
  
  state.feedback_pending = false;
  stats.feedbacks_received++;
  
  Serial.println("✓ Prediction accepted and added to learning buffer");
}

void rejectPrediction(int user_brightness) {
  if (!state.feedback_pending) return;
  
  // User provided different brightness
  setBrightness(user_brightness);
  state.actual_brightness = user_brightness;
  
  // Add corrected data to learning buffer
  addToLearningBuffer(
    state.ambient_light,
    state.motion_detected,
    state.hour,
    state.day_of_week,
    user_brightness
  );
  
  // Update error statistics
  int error = abs(state.predicted_brightness - user_brightness);
  stats.avg_prediction_error = (stats.avg_prediction_error * stats.feedbacks_received + error) 
                                / (stats.feedbacks_received + 1);
  
  state.feedback_pending = false;
  stats.feedbacks_received++;
  
  Serial.printf("✓ User feedback: %d%% (error: %d%%)\n", 
                user_brightness, error);
}

// ============================================================================
// ONLINE LEARNING FUNCTIONS
// ============================================================================

void addToLearningBuffer(float ambient, int motion, int hour, 
                        int day_of_week, int brightness) {
  // Calculate time features
  float sin_h, cos_h;
  int time_period;
  calculate_time_features(hour, &sin_h, &cos_h, &time_period);
  
  // Add to buffer
  learningBuffer[bufferIndex] = {
    ambient, motion, sin_h, cos_h, time_period, day_of_week, 
    brightness, millis()
  };
  
  bufferIndex = (bufferIndex + 1) % LEARNING_BUFFER_SIZE;
  if (bufferCount < LEARNING_BUFFER_SIZE) bufferCount++;
  
  Serial.printf("Learning buffer: %d/%d samples\n", 
                bufferCount, LEARNING_BUFFER_SIZE);
  
  // Auto-save periodically
  static int save_counter = 0;
  if (++save_counter >= 10) {
    saveLearningBuffer();
    save_counter = 0;
  }
}

void saveLearningBuffer() {
  File file = SPIFFS.open("/learning_data.csv", "w");
  if (!file) {
    Serial.println("ERROR: Failed to open learning data file for writing");
    return;
  }
  
  // Write header
  file.println("ambient_light,motion_detected,sin_hour,cos_hour,time_period,day_of_week,led_brightness,timestamp");
  
  // Write data
  for (int i = 0; i < bufferCount; i++) {
    LearningData& data = learningBuffer[i];
    file.printf("%.2f,%d,%.4f,%.4f,%d,%d,%d,%lu\n",
                data.ambient_light, data.motion_detected,
                data.sin_hour, data.cos_hour,
                data.time_period, data.day_of_week,
                data.brightness, data.timestamp);
  }
  
  file.close();
  Serial.printf("✓ Learning buffer saved (%d samples)\n", bufferCount);
}

void loadLearningBuffer() {
  File file = SPIFFS.open("/learning_data.csv", "r");
  if (!file) {
    Serial.println("No existing learning data found");
    return;
  }
  
  // Skip header
  file.readStringUntil('\n');
  
  bufferCount = 0;
  bufferIndex = 0;
  
  while (file.available() && bufferCount < LEARNING_BUFFER_SIZE) {
    String line = file.readStringUntil('\n');
    if (line.length() > 0) {
      // Parse CSV line
      sscanf(line.c_str(), "%f,%d,%f,%f,%d,%d,%d,%lu",
             &learningBuffer[bufferCount].ambient_light,
             &learningBuffer[bufferCount].motion_detected,
             &learningBuffer[bufferCount].sin_hour,
             &learningBuffer[bufferCount].cos_hour,
             &learningBuffer[bufferCount].time_period,
             &learningBuffer[bufferCount].day_of_week,
             &learningBuffer[bufferCount].brightness,
             &learningBuffer[bufferCount].timestamp);
      
      bufferCount++;
      bufferIndex = bufferCount % LEARNING_BUFFER_SIZE;
    }
  }
  
  file.close();
  Serial.printf("✓ Loaded %d samples from learning buffer\n", bufferCount);
}

void exportLearningData() {
  // Export learning data in format ready for retraining
  File file = SPIFFS.open("/export_training_data.csv", "w");
  if (!file) {
    Serial.println("ERROR: Failed to export learning data");
    return;
  }
  
  file.println("ambient_light,motion_detected,sin_hour,cos_hour,time_period,day_of_week,led_brightness");
  
  for (int i = 0; i < bufferCount; i++) {
    LearningData& data = learningBuffer[i];
    file.printf("%.2f,%d,%.4f,%.4f,%d,%d,%d\n",
                data.ambient_light, data.motion_detected,
                data.sin_hour, data.cos_hour,
                data.time_period, data.day_of_week,
                data.brightness);
  }
  
  file.close();
  Serial.println("✓ Learning data exported for retraining");
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
  server.on("/api/export", HTTP_GET, handleExport);
  server.on("/api/stats", HTTP_GET, handleStats);
  
  Serial.println("✓ Web server routes configured");
}

void handleRoot() {
  String html = R"(
<!DOCTYPE html>
<html>
<head>
  <title>ESP32 Adaptive Lighting</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial; margin: 20px; background: #f0f0f0; }
    .container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #333; text-align: center; }
    .status { padding: 15px; margin: 10px 0; background: #e8f4f8; border-radius: 5px; }
    .status-item { margin: 8px 0; }
    .label { font-weight: bold; color: #555; }
    .value { color: #007bff; }
    button { padding: 12px 24px; margin: 5px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; }
    .btn-accept { background: #28a745; color: white; }
    .btn-reject { background: #dc3545; color: white; }
    .btn-manual { background: #007bff; color: white; }
    .slider { width: 100%; margin: 10px 0; }
    #brightness-value { font-size: 24px; color: #007bff; font-weight: bold; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🔆 Adaptive Lighting Control</h1>
    
    <div class="status">
      <h2>Current Status</h2>
      <div class="status-item"><span class="label">Ambient Light:</span> <span class="value" id="ambient">--</span> lux</div>
      <div class="status-item"><span class="label">Motion:</span> <span class="value" id="motion">--</span></div>
      <div class="status-item"><span class="label">Time:</span> <span class="value" id="time">--</span></div>
      <div class="status-item"><span class="label">Predicted:</span> <span class="value" id="predicted">--</span>%</div>
      <div class="status-item"><span class="label">Current:</span> <span class="value" id="current">--</span>%</div>
    </div>
    
    <div class="status">
      <h2>Feedback</h2>
      <p>Is the brightness correct?</p>
      <button class="btn-accept" onclick="sendFeedback('accept')">✓ Accept</button>
      <button class="btn-reject" onclick="sendFeedback('reject')">✗ Adjust</button>
    </div>
    
    <div class="status">
      <h2>Manual Control</h2>
      <input type="range" min="0" max="100" value="50" class="slider" id="brightness-slider" oninput="updateSlider(this.value)">
      <p><span id="brightness-value">50</span>%</p>
      <button class="btn-manual" onclick="sendManual()">Set Brightness</button>
    </div>
    
    <div class="status">
      <h2>Statistics</h2>
      <div class="status-item"><span class="label">Predictions:</span> <span class="value" id="predictions">--</span></div>
      <div class="status-item"><span class="label">Feedbacks:</span> <span class="value" id="feedbacks">--</span></div>
      <div class="status-item"><span class="label">Learning Buffer:</span> <span class="value" id="buffer">--</span></div>
      <div class="status-item"><span class="label">Avg Error:</span> <span class="value" id="error">--</span>%</div>
    </div>
    
    <div style="text-align: center; margin-top: 20px;">
      <button class="btn-manual" onclick="exportData()">📥 Export Learning Data</button>
    </div>
  </div>
  
  <script>
    function updateStatus() {
      fetch('/api/status')
        .then(r => r.json())
        .then(data => {
          document.getElementById('ambient').textContent = data.ambient_light.toFixed(0);
          document.getElementById('motion').textContent = data.motion_detected ? 'Detected' : 'None';
          document.getElementById('time').textContent = data.hour + ':00, Day ' + data.day_of_week;
          document.getElementById('predicted').textContent = data.predicted_brightness;
          document.getElementById('current').textContent = data.actual_brightness;
        });
      
      fetch('/api/stats')
        .then(r => r.json())
        .then(data => {
          document.getElementById('predictions').textContent = data.predictions_made;
          document.getElementById('feedbacks').textContent = data.feedbacks_received;
          document.getElementById('buffer').textContent = data.buffer_count + '/' + data.buffer_size;
          document.getElementById('error').textContent = data.avg_error.toFixed(1);
        });
    }
    
    function sendFeedback(action) {
      let brightness = action === 'reject' ? document.getElementById('brightness-slider').value : null;
      fetch('/api/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action: action, brightness: brightness})
      }).then(() => alert('Feedback sent!'));
    }
    
    function sendManual() {
      let brightness = document.getElementById('brightness-slider').value;
      fetch('/api/manual', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({brightness: parseInt(brightness)})
      }).then(() => alert('Brightness set to ' + brightness + '%'));
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
  StaticJsonDocument<256> doc;
  doc["ambient_light"] = state.ambient_light;
  doc["motion_detected"] = state.motion_detected;
  doc["hour"] = state.hour;
  doc["day_of_week"] = state.day_of_week;
  doc["predicted_brightness"] = state.predicted_brightness;
  doc["actual_brightness"] = state.actual_brightness;
  doc["feedback_pending"] = state.feedback_pending;
  
  String json;
  serializeJson(doc, json);
  server.send(200, "application/json", json);
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
  state.actual_brightness = brightness;
  
  // Add to learning buffer
  addToLearningBuffer(
    state.ambient_light,
    state.motion_detected,
    state.hour,
    state.day_of_week,
    brightness
  );
  
  server.send(200, "text/plain", "Manual brightness set");
}

void handleExport() {
  exportLearningData();
  
  File file = SPIFFS.open("/export_training_data.csv", "r");
  if (!file) {
    server.send(500, "text/plain", "Export failed");
    return;
  }
  
  server.streamFile(file, "text/csv");
  file.close();
}

void handleStats() {
  StaticJsonDocument<256> doc;
  doc["predictions_made"] = stats.predictions_made;
  doc["feedbacks_received"] = stats.feedbacks_received;
  doc["avg_error"] = stats.avg_prediction_error;
  doc["buffer_count"] = bufferCount;
  doc["buffer_size"] = LEARNING_BUFFER_SIZE;
  doc["uptime"] = millis() / 1000;
  
  String json;
  serializeJson(doc, json);
  server.send(200, "application/json", json);
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
