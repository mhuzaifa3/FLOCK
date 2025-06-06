import time
from RPi.GPIO as GPIO
from faceRecognition import facerec
import time

GPIO.setup(11, GPIO.IN)

while True:
    i = GPIO.input(11)
    if i == 0:
        time.sleep(0.1)
    elif i == 1:
        print("Motion detected")
        facerec()
