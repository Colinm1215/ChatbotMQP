import queue
import time

import serial


print("Opening serial port...")
ser = serial.Serial('COM3', 115200)
time.sleep(5)
print("Serial port opened.")
print("{500,R}".encode())
ser.write("{500,R}".encode())
ser.flush()
print("Command sent to Arduino.")
time.sleep(0.1)
ser.close()
print("Serial port closed.")



