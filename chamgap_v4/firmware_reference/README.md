# Firmware references — not production firmware

`src/sensor_bringup.ino` only collects raw ADC/HX711 and paired temperature/humidity to Serial.
It NEVER sets pump GPIO HIGH. Use it with motor power physically disconnected.
Install the Espressif Arduino core, Adafruit SHT31 and bogde HX711; select the actual board.
The Arduino-specific sketch has not been compiled or run on ESP32 in this environment.

`include/pulse_guard.h` is a pure C++ timer/interlock logic reference. A native compiler can
check arithmetic including millis() rollover using `tests_cpp/pulse_guard_test.cpp`.
It does not implement persistent budgets, command authentication, watchdogs, clock leases,
calibrated dosing, or certified emergency switching. Do not connect it directly to motors.

Production firmware must implement the full command and telemetry contracts in the master MD.
