import serial
import fire
import time
from datetime import datetime
import sys, os
from config import kwargs_rest
import io,re
import cv2
import stapipy as st
import numpy as np
from auto_functions_opencv import CMyCallback
sys.path.append(os.getcwd())
from image_processing import process_image
from predict import detect
# from pneumatic_control import send_signal_job

# initialize for arduino + camera
ser = serial.Serial(port='COM6', baudrate=9600)
ser_nano = serial.Serial(port='COM3', baudrate=9600)
time.sleep(3)

# Initialization
st.initialize()
st_system = st.create_system()
st_device = st_system.create_first_device()
st_datastream = st_device.create_datastream()
st_datastream.start_acquisition()
st_device.acquisition_start()

my_callback = CMyCallback()
cb_func = my_callback.datastream_callback
# Register callback for datastream
callback = st_datastream.register_callback(cb_func)

FolderName = "Testing-7Jan2023"
EXPOSURE_MODE = "ExposureMode"
EXPOSURE_TIME = "ExposureTime"
EXPOSURE_TIME_RAW = "ExposureTimeRaw" # Custom
responses = {}



def turnOn():
    num = 0
    capture = False
    response = None
    start_read = None
    res = 0
    run = True

    fraksi_dic = {'1': 'MENTAH', '2': 'MATANG'}
    skenario_dic = {'1': 'D', '2': 'B'}
    responses = {}    # sampel = 0

    sampel = datetime.now().strftime("sampel_%d-%m-%y-%H:%M:%S")
    intensity = []

    while run:
        acquisition_command = ser_nano.readline().decode()
        print(f"acquisition_command: {acquisition_command}")

        kwargs_rest['key_input'] = cv2.waitKey(1)
        # check if conveyor stopped
        if "berhenti" in acquisition_command:
            print('conveyor stop. starting to acquisition')
            time.sleep(1)
            kwargs_rest['status_gerak'] = True

        if kwargs_rest['key_input'] == ord('q'):
            break 

        output_image = my_callback.image
        # output_image = cv2.resize(output_image, (1024, 540))
        if output_image is not None:
            cv2.imshow('frame', output_image)
            output_image = output_image

        if kwargs_rest["status_gerak"] == True:
            # check the folder created
            if not os.path.exists(FolderName):
                os.makedirs(FolderName)
            
            print(f'kwargs_rest["status_gerak"]: {kwargs_rest["status_gerak"]}')
            num += 1
            print(f'num = {num}..............................................................................................')

        if num >= 1 and kwargs_rest["status_gerak"]:
            print(f"Activate the LED lamp: {num}")
            start_time = time.time()
            ser.write(str(num).encode())
            response = ser.readline().decode()
            print(f'response from led camera: {response}..............................................................................................')
            # if "ready" in response and start_read is None:
            #     start_read = True
            # if start_read is None:
            #     start_read = True
            # print(f"start_read: {start_read}")

        # if start_read and kwargs_rest["status_gerak"]:
        if kwargs_rest["status_gerak"]:
            try:
                res = int(response)
            except:
                continue
            print(f"Response after LED lamp ON: {res}")

            # cv2.imwrite(f"{FolderName}/{fraksi}_S{sampel}_LED{res}_{skenario}.png", output_image)
            cv2.imwrite(f"{FolderName}/{sampel}-LED{res}.png", output_image)
            print(f"image file: {FolderName}/{sampel}-LED{res}.png, Succesfully Saved!!")
            intensity_mean = process_image(output_image)
            intensity.append(intensity_mean)

            if res == 9:
                # send to turn off the led lamp
                print("turn off the led lamp")
                ser.write(str(1).encode())

        if num == 9 and kwargs_rest["status_gerak"]:
            print(f"Sending command to turn on led is done | no_led: {num}")
            num = 0

        if res == 9 and kwargs_rest["status_gerak"]:
            print("Image acquisition is done.")
            result = detect(np.array(intensity[-8:]).reshape(1, -1))
            # send to api to display graph and image (there are 8 images for each fruit)
            print(f"Prediction Result: {result}")

            if result["result"] == 'MENTAH':
                # activate pneumatic to throw the MENTAH fruit
                ser_nano.write(str(2).encode())
            else:
                # re-run conveyor. do nothing with MATANG level
                ser_nano.write(str(3).encode())
            end_time = time.time() - start_time
            print(f"elapsed time to acquisition: {end_time}s.")
            responses = None
            num = 0
            res = 0
            # start_read = None
            kwargs_rest["status_gerak"] = False
            sampel = datetime.now().strftime("sampel_%d-%m-%y-%H:%M:%S")
            print(f"Image Processind is done.")
            # return turnOn
            # run = True
            # time.sleep(1)
            # print("Waiting 1s to stop.")

    st_device.acquisition_stop()           
    st_datastream.stop_acquisition()


if __name__ == '__main__':
	fire.Fire()
 