#pragma once
// GPIO numbers for the classic ESP32 ONLY. Verify actual board silkscreen.
namespace pins {
constexpr int SDA_PIN=21, SCL_PIN=22;
constexpr int SOIL_A=32, SOIL_B=33;
constexpr int HX_DT=26, HX_SCK=27;
constexpr int PUMP_SIG=25;
constexpr int ESTOP_AUX=34, LEAK_AUX=35; // input-only, require EXTERNAL bias
}
