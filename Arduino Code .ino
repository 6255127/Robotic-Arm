#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ── PCA9685 ──────────────────────────────────────────────────────────────────
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// ── Physical channel mapping  ─────────────────────────────────────────────────
// CHANGE these numbers if a servo responds to the wrong slider.
// Each value is the PCA9685 output channel (0-15) for that joint.
const uint8_t CH[5] = {
  0,   // joint 0 → Base     → PCA9685 channel 0
  1,   // joint 1 → Shoulder → PCA9685 channel 1
  2,   // joint 2 → Elbow    → PCA9685 channel 2
  3,   // joint 3 → Wrist    → PCA9685 channel 3
  4,   // joint 4 → Gripper  → PCA9685 channel 4
};

// ── Servo pulse limits (µs) — tune to your servos ────────────────────────────
#define SERVO_MIN_US  500
#define SERVO_MAX_US  2400

// ── Gripper safety ────────────────────────────────────────────────────────────
#define GRIP_MIN  20
#define GRIP_MAX  110

// ── Smooth-motion parameters ──────────────────────────────────────────────────
#define STEP_MS       15      // servo update interval (ms)
#define SMOOTH        0.18f   // exponential factor per step (0.1=slow, 0.3=fast)

// ── State ─────────────────────────────────────────────────────────────────────
int   target[5]  = {90, 90, 90, 90, 70};
float current[5] = {90, 90, 90, 90, 70};
unsigned long lastStep = 0;
String inputBuf = "";

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  Wire.begin();
  Wire.setClock(400000);   // fast I2C — essential for reliable 5-channel writes

  pwm.begin();
  pwm.setOscillatorFrequency(25000000);
  pwm.setPWMFreq(50);      // 50 Hz standard servo frequency
  delay(100);

  // Write home position to all servos
  for (int i = 0; i < 5; i++) writeServo(i, target[i]);

  Serial.println(F("[BOOT] RoboArm Pro v2 — non-blocking firmware ready"));
  Serial.println(F("[BOOT] Format: base,shoulder,elbow,wrist,gripper"));
}

// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  // ── 1. Non-blocking servo update ─────────────────────────────────────────
  unsigned long now = millis();
  if (now - lastStep >= STEP_MS) {
    lastStep = now;
    smoothUpdate();
  }

  // ── 2. Serial receive — ALWAYS responsive (never blocked by motion) ───────
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      inputBuf.trim();
      if (inputBuf.length() > 0) {
        parseAndApply(inputBuf);
        inputBuf = "";
      }
    } else if (inputBuf.length() < 80) {
      inputBuf += c;
    }
  }
}

// ── Exponential-easing update (called every STEP_MS) ─────────────────────────
void smoothUpdate() {
  for (int i = 0; i < 5; i++) {
    float diff = (float)target[i] - current[i];
    if (fabsf(diff) > 0.12f) {
      current[i] += diff * SMOOTH;
    } else {
      current[i] = (float)target[i];
    }
    writeServo(i, (int)roundf(current[i]));
  }
}

// ── Parse "b,s,e,w,g" and update targets immediately ─────────────────────────
void parseAndApply(const String& cmd) {
  int vals[5];
  int count = 0;
  int start = 0;

  for (int i = 0; i <= (int)cmd.length() && count < 5; i++) {
    if (i == (int)cmd.length() || cmd.charAt(i) == ',') {
      vals[count++] = cmd.substring(start, i).toInt();
      start = i + 1;
    }
  }

  if (count != 5) {
    Serial.print(F("[ERR] Need 5 values, got "));
    Serial.println(count);
    return;
  }

  target[0] = constrain(vals[0], 0,        180);
  target[1] = constrain(vals[1], 0,        180);
  target[2] = constrain(vals[2], 0,        180);
  target[3] = constrain(vals[3], 0,        180);
  target[4] = constrain(vals[4], GRIP_MIN, GRIP_MAX);

  Serial.print(F("[OK] B=")); Serial.print(target[0]);
  Serial.print(F(" S="));    Serial.print(target[1]);
  Serial.print(F(" E="));    Serial.print(target[2]);
  Serial.print(F(" W="));    Serial.print(target[3]);
  Serial.print(F(" G="));    Serial.println(target[4]);
}

// ── Write one joint angle to its PCA9685 channel ─────────────────────────────
void writeServo(int joint, int degrees) {
  degrees = constrain(degrees, 0, 180);
  if (joint == 4) {
    degrees = constrain(degrees, GRIP_MIN, GRIP_MAX);
  }
  uint16_t us = (uint16_t)map(degrees, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  pwm.writeMicroseconds(CH[joint], us);
}

