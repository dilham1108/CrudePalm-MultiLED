import serial
import time
import glob
from datetime import datetime

def get_arduino_port():
    ports = glob.glob("/dev/ttyACM*")
    if not ports:
        return None

    return ports[0]


def send_command(payload):
    
    arduino_port = get_arduino_port()

    if not arduino_port:
        print("> arduino not found")
        return

    arduino = serial.Serial(port=arduino_port, baudrate=9600, timeout=.1)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"> sending command: 1 for payload {payload} at {now}")
    arduino.write(bytes('2', 'utf-8'))
    data = arduino.readline()
    return data


if __name__ == "__main__":
    import fire
    fire.Fire()
