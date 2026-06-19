from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class SensorReader:
    simulate: bool = True
    dht_pin: int = 4
    pir_pin: int = 3

    def __post_init__(self):
        self._dht = None
        self._pir = None
        if not self.simulate:
            try:
                import adafruit_dht
                import board
                from gpiozero import MotionSensor

                dht_board_pin = getattr(board, f"D{self.dht_pin}")
                self._dht = adafruit_dht.DHT22(dht_board_pin, use_pulseio=False)
                self._pir = MotionSensor(self.pir_pin)
            except Exception as exc:
                raise RuntimeError("Hardware sensor initialization failed; set SIMULATE_SENSORS=true for testing") from exc

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def read(self) -> dict[str, Any]:
        timestamp = self._timestamp()
        if self.simulate:
            temperature = round(random.uniform(19.0, 27.0), 1)
            humidity = round(random.uniform(38.0, 68.0), 1)
            motion = int(random.random() > 0.25)
        else:
            try:
                temperature = float(self._dht.temperature)
                humidity = float(self._dht.humidity)
            except RuntimeError:
                time.sleep(0.2)
                temperature = float(self._dht.temperature)
                humidity = float(self._dht.humidity)
            motion = int(bool(self._pir.motion_detected))
        return {
            "dht22": {
                "gpio_pin": self.dht_pin,
                "timestamp": timestamp,
                "temperature_celsius": temperature,
                "humidity_percent": humidity,
            },
            "PIR501": {"value": motion, "gpio_pin": self.pir_pin, "timestamp": timestamp},
        }
