/*
 * utility_agent.h
 *
 * NON-ML AI LAYER: a Utility-Based Intelligent Agent (Russell & Norvig style).
 *
 * This is NOT machine learning. Nothing here is trained or learned from data.
 * It is classical deliberative AI: CONSTRAINED UTILITY MAXIMIZATION. The agent
 * searches candidate brightness levels and chooses the one that maximizes a
 * hand-designed utility function, subject to a hard safety constraint.
 *
 *   Soft objective (traded off):
 *     U(b) =  w_comfort * Comfort(b)     // closeness to the RF-KNN target
 *           - w_energy  * Energy(b)      // power cost (higher b = costlier)
 *           - w_smooth  * Smoothness(b)  // discourage abrupt jumps
 *
 *   Hard constraint (guaranteed, never optimized away):
 *     if night AND motion -> b >= min_safe_night     // guaranteed visibility
 *
 * The agent does NOT modify the RF-KNN pipeline. It only READS:
 *   - ml_brightness : the RF-KNN output, used purely as the "comfort anchor"
 *   - the same sensor inputs (ambient, motion) and the clock (hour)
 * and produces its own recommendation plus a per-term breakdown.
 *
 * Decision rule: argmax over the feasible candidate brightness range, step
 * `step`. This brute-force search over the action space is classic AI
 * optimization (no gradients, no training).
 *
 * Config (weights / thresholds) lives in RAM with hand-picked defaults and is
 * adjustable at runtime via the `setutil` serial command. The EEPROM layout
 * used by the RF-KNN system is intentionally NOT touched here.
 */

#ifndef UTILITY_AGENT_H
#define UTILITY_AGENT_H

#include <Arduino.h>
#include <math.h>

// --------------------------------------------
// Tunable configuration (hand-designed, NOT learned)
// --------------------------------------------
struct UtilityConfig {
  float w_comfort;       // weight: stay close to RF-KNN target
  float w_energy;        // weight: penalize brightness (power)
  float w_smooth;        // weight: penalize big jumps from previous level
  int   min_safe_night;  // HARD floor (%): min brightness when night AND motion
  int   night_start_hour;// night begins at/after this hour ...
  int   night_end_hour;  // ... and ends before this hour (wrap-around ok)
  int   step;            // candidate brightness step for the argmax search
};

// Gentle defaults: track the RF-KNN target closely, with a light energy nudge,
// mild smoothing, and a firm 40% visibility floor on night-time motion.
UtilityConfig g_util_cfg = {
  /* w_comfort        */ 1.0f,
  /* w_energy         */ 0.3f,
  /* w_smooth         */ 0.3f,
  /* min_safe_night   */ 40,
  /* night_start_hour */ 21,
  /* night_end_hour   */ 6,
  /* step             */ 5
};

// --------------------------------------------
// Result / breakdown of one evaluation
// --------------------------------------------
struct UtilityResult {
  bool  valid;
  float ml_anchor;          // RF-KNN brightness used as the comfort anchor
  float prev_brightness;    // previous applied level (for the smoothness term)
  float ambient;            // ambient lux at evaluation time
  int   motion;             // motion flag at evaluation time
  int   hour;               // hour of day
  bool  is_night;           // night gate result
  bool  safety_floor_active;// true when the hard floor was enforced
  int   safety_floor;       // the floor value that applied (0 if inactive)
  float recommended;        // argmax brightness chosen by the agent
  float utility;            // U at the recommended level
  // signed soft-term contributions AT the recommended level (already weighted):
  float comfort_term;       // +
  float energy_term;        // -
  float smooth_term;        // -
};

// --------------------------------------------
// Night gate (supports wrap-around, e.g. 21:00..06:00)
// --------------------------------------------
static inline bool utilIsNight(int hour, const UtilityConfig &c) {
  if (c.night_start_hour <= c.night_end_hour) {
    return (hour >= c.night_start_hour && hour < c.night_end_hour);
  }
  return (hour >= c.night_start_hour || hour < c.night_end_hour);
}

// --------------------------------------------
// Individual soft utility terms (RAW, before weighting)
// --------------------------------------------

// Comfort peaks at 1.0 when b == anchor, falls off quadratically.
static inline float utilComfortRaw(float b, float anchor) {
  float d = (b - anchor) / 100.0f;
  return 1.0f - d * d;
}

// Energy cost in [0..1] (linear in brightness).
static inline float utilEnergyRaw(float b) {
  return b / 100.0f;
}

// Smoothness penalty: quadratic in the jump from the previous level.
static inline float utilSmoothRaw(float b, float prev) {
  float d = (b - prev) / 100.0f;
  return d * d;
}

// Total (weighted) SOFT utility for a candidate brightness b.
static inline float utilTotal(float b, float anchor, float prev,
                             const UtilityConfig &c) {
  return  c.w_comfort * utilComfortRaw(b, anchor)
        - c.w_energy  * utilEnergyRaw(b)
        - c.w_smooth  * utilSmoothRaw(b, prev);
}

// --------------------------------------------
// Evaluate: constrained argmax over the feasible action range
// Read-only; fills `out` with the recommendation + breakdown.
// --------------------------------------------
static inline void utilityAgentEvaluate(float ml_brightness, float prev_brightness,
                                        float ambient, int motion, int hour,
                                        UtilityResult *out) {
  const UtilityConfig &c = g_util_cfg;
  bool  night = utilIsNight(hour, c);
  int   step  = (c.step >= 1) ? c.step : 5;

  // HARD safety constraint: lower-bound the feasible range.
  bool  floor_active = (night && motion == 1);
  int   lo = 0;
  if (floor_active) {
    lo = c.min_safe_night;
    if (lo < 0)   lo = 0;
    if (lo > 100) lo = 100;
  }

  float bestB = (float)lo;            // default to the feasible lower bound
  float bestU = -INFINITY;

  for (int b = lo; b <= 100; b += step) {
    float u = utilTotal((float)b, ml_brightness, prev_brightness, c);
    if (u > bestU) { bestU = u; bestB = (float)b; }
  }

  if (out) {
    out->valid               = true;
    out->ml_anchor           = ml_brightness;
    out->prev_brightness     = prev_brightness;
    out->ambient             = ambient;
    out->motion              = motion;
    out->hour                = hour;
    out->is_night            = night;
    out->safety_floor_active = floor_active;
    out->safety_floor        = floor_active ? lo : 0;
    out->recommended         = bestB;
    out->utility             = bestU;
    out->comfort_term        =  c.w_comfort * utilComfortRaw(bestB, ml_brightness);
    out->energy_term         = -c.w_energy  * utilEnergyRaw(bestB);
    out->smooth_term         = -c.w_smooth  * utilSmoothRaw(bestB, prev_brightness);
  }
}

// --------------------------------------------
// Pretty-print the last evaluation (for the `utility` serial command)
// --------------------------------------------
static inline void utilityAgentPrint(const UtilityResult &r) {
  Serial.println();
  Serial.println(F("=== UTILITY-BASED AGENT (non-ML AI) ==="));
  if (!r.valid) {
    Serial.println(F("No evaluation yet (wait for the next live update)."));
    Serial.println();
    return;
  }

  Serial.print(F("  Inputs : ML anchor=")); Serial.print(r.ml_anchor, 1);
  Serial.print(F("%  prev="));              Serial.print(r.prev_brightness, 1);
  Serial.print(F("%  ambient="));           Serial.print(r.ambient, 1);
  Serial.print(F("lux  motion="));          Serial.print(r.motion);
  Serial.print(F("  hour="));               Serial.print(r.hour);
  Serial.print(F("  ("));                   Serial.print(r.is_night ? F("night") : F("day"));
  Serial.println(F(")"));

  Serial.print(F("  Weights: comfort="));   Serial.print(g_util_cfg.w_comfort, 2);
  Serial.print(F(" energy="));              Serial.print(g_util_cfg.w_energy, 2);
  Serial.print(F(" smooth="));              Serial.println(g_util_cfg.w_smooth, 2);

  Serial.print(F("  Safety floor: "));
  if (r.safety_floor_active) {
    Serial.print(F("ACTIVE (>= ")); Serial.print(r.safety_floor); Serial.println(F("% : night + motion)"));
  } else {
    Serial.println(F("inactive"));
  }

  Serial.println(F("  Soft-term contributions at recommended level:"));
  Serial.print(F("    + comfort : ")); Serial.println(r.comfort_term, 3);
  Serial.print(F("    - energy  : ")); Serial.println(r.energy_term, 3);
  Serial.print(F("    - smooth  : ")); Serial.println(r.smooth_term, 3);
  Serial.print(F("    = utility : ")); Serial.println(r.utility, 3);

  Serial.print(F("  Recommended brightness : ")); Serial.print(r.recommended, 1);
  Serial.println(F("%"));
  Serial.print(F("  RF-KNN anchor was       : ")); Serial.print(r.ml_anchor, 1);
  Serial.println(F("%"));
  Serial.println();
}

#endif // UTILITY_AGENT_H
