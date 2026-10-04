/* SENSOR-ONLY commissioning sketch. Pump power MUST remain disconnected.
 * Uses Adafruit SHT31 and bogde HX711 libraries; actual ESP32 build/board not tested here.
 * This is not the MQTT production firmware. Serial output uses a separate bringup schema.
 * No mass calibration: HX711 outputs raw counts only. No soil % conversion.
 */
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_SHT31.h>
#include <HX711.h>
#include "../include/pins.h"
Adafruit_SHT31 airA,airB;
HX711 scale;
bool okA=false,okB=false;
uint32_t lastSample=0;

void setup(){
  digitalWrite(pins::PUMP_SIG,LOW);
  pinMode(pins::PUMP_SIG,OUTPUT); // Hardware external pulldown still required.
  pinMode(pins::ESTOP_AUX,INPUT);
  pinMode(pins::LEAK_AUX,INPUT);
  Serial.begin(115200);
  Wire.begin(pins::SDA_PIN,pins::SCL_PIN);
  Wire.setClock(100000);
  analogReadResolution(12);
  okA=airA.begin(0x44);
  okB=airB.begin(0x45); // Configure ADDR on the ACTUAL second module first.
  scale.begin(pins::HX_DT,pins::HX_SCK);
  Serial.println("bringup.csv.v1,uptime_ms,tA_c,rhA_pct,tB_c,rhB_pct,soilA_adc,soilB_adc,hx_raw,motor_enabled");
}
void printFinite(float v){if(isfinite(v))Serial.print(v,3);else Serial.print("NA");}
void loop(){
  digitalWrite(pins::PUMP_SIG,LOW); // No code path sets HIGH in this sketch.
  uint32_t now=millis();
  if(uint32_t(now-lastSample)<2000)return;
  lastSample=now;
  float ta=okA?airA.readTemperature():NAN;
  float ha=okA?airA.readHumidity():NAN;
  float tb=okB?airB.readTemperature():NAN;
  float hb=okB?airB.readHumidity():NAN;
  int sa=analogRead(pins::SOIL_A),sb=analogRead(pins::SOIL_B);
  Serial.print("bringup.csv.v1,");Serial.print(now);Serial.print(',');
  printFinite(ta);Serial.print(',');printFinite(ha);Serial.print(',');
  printFinite(tb);Serial.print(',');printFinite(hb);Serial.print(',');
  Serial.print(sa);Serial.print(',');Serial.print(sb);Serial.print(',');
  if(scale.is_ready())Serial.print(scale.read());else Serial.print("NA");
  Serial.println(",false");
}
