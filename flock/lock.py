"""Lock actuation, with the GPIO dependency isolated behind an interface."""
from __future__ import annotations

import logging
import time
from typing import Protocol

logger = logging.getLogger(__name__)


class Lock(Protocol):
    def unlock(self, seconds: float) -> None: ...
    def is_locked(self) -> bool: ...


class SimulatedLock:
    """Used off-device, including in tests, so nothing needs a Raspberry Pi."""

    def __init__(self) -> None:
        self._locked = True
        self.unlock_calls: list[float] = []

    def unlock(self, seconds: float) -> None:
        self._locked = False
        self.unlock_calls.append(seconds)
        logger.info("unlocked for %.1fs", seconds)
        self._locked = True

    def is_locked(self) -> bool:
        return self._locked


class GpioLock:
    def __init__(self, pin: int = 18, active_high: bool = True) -> None:
        import RPi.GPIO as GPIO

        self._gpio = GPIO
        self._pin = pin
        self._active_high = active_high
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, not active_high)
        self._locked = True

    def unlock(self, seconds: float) -> None:
        self._gpio.output(self._pin, self._active_high)
        self._locked = False
        time.sleep(seconds)
        self._gpio.output(self._pin, not self._active_high)
        self._locked = True

    def is_locked(self) -> bool:
        return self._locked


def default_lock(pin: int = 18) -> Lock:
    try:
        return GpioLock(pin=pin)
    except (ImportError, RuntimeError):
        logger.info("no GPIO available, using simulated lock")
        return SimulatedLock()
