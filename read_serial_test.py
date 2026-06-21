import serial

PORT = "/dev/cu.usbmodem101" # port prüfen mit: ls /dev/cu.*
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Lese Daten vom ESP32...")
print("Abbrechen mit CTRL + C")

while True:
    line = ser.readline().decode(errors="ignore").strip()

    if line:
        print(line)