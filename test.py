import serial
import fire, sys, os
import time
import cv2
sys.path.append(os.getcwd())

ser_ir = serial.Serial(port='COM3', baudrate=9600)
time.sleep(3)

def turnOn():
    run = True

    while run:
        res_ir = ser_ir.readline().decode()
        print(f"Response from infrared: {res_ir}")
        if res_ir[:6] == "BUAH":
            print("Found a object")
        else:
            print("Waiting for the next object...!")


if __name__ == '__main__':
	fire.Fire()
