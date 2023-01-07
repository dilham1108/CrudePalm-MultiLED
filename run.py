import serial
import fire
import time
import sys, os
from config import kwargs_rest
import io,re
import cv2
import stapipy as st
import numpy as np
from datetime import datetime
from auto_functions_opencv import CMyCallback
sys.path.append(os.getcwd())

from image_processing import process_image
from predict import jst_detect
from pneumatic_control import send_signal_job

# initialize for arduino + camera
ser = serial.Serial(port='COM6', baudrate=9600)
time.sleep(2)

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

FolderName = "MultispektralLED"
EXPOSURE_MODE = "ExposureMode"
EXPOSURE_TIME = "ExposureTime"
EXPOSURE_TIME_RAW = "ExposureTimeRaw" # Custom
responses = {}
URL = "http://localhost:6000/graph"
code_pneumatic = 2

def turnOn():
    run = True
    start_time = None
    responses = {}    # sampel = 0

    start_system = input('Start? (y/n): ')
    intensity = []

    while run:
        kwargs_rest['key_input'] = cv2.waitKey(1)
        if start_system == "y":
            kwargs_rest['responses']['moved'] = '1'
            # turn on all leds lamp
            ser.write(b"0")
            time.sleep(2)

        else:
            capture = False
            print("Exit...!")
            kwargs_rest['responses']['moved'] = '0'
            # turn off all leds
            ser.write(b'1')
            break

        if kwargs_rest['key_input'] == ord('q'):
            run = False
            # turn off all leds
            ser.write(b'1')  
            break 

        output_image = my_callback.image
        if output_image is not None:
            cv2.imshow('frame', output_image)
        else:
            continue

        if kwargs_rest["responses"]['moved'] == '1':
            kwargs_rest["status_gerak"] = True
            # check if folder created
            if not os.path.exists(FolderName):
                os.makedirs(FolderName)

        if kwargs_rest["status_gerak"]:
            # process frame to get the intensity mean. this result will be used to detect alb and kadar minyak
            intensity_mean = process_image(output_image)
            print(f"intensity_mean: {intensity_mean}")
            # todo: detect result of alb and oil level
            result = jst_detect(np.array(intensity[-8:]).reshape(1, -1))

            # todo: update yolo here and put the result of alb and oil level into bounding box


        # if num > 1 and acquire == True:
        #     start_time = time.time()
        #     ser.write(str(num).encode())
        #     response = ser.readline().decode()
        #     if "ready" in response and start_read is None:
        #         start_read = True
        # if num > 1 and start_read and acquire == True:
        #     try:
        #         res = int(response)
        #     except:
        #         continue

        #     if res <= prev_res:
        #         continue

        #     if res >= 2 and res <=9:
        #         print(f"response from arduino: {res}")
        #         imagefile = f"{FolderName}/LED{res}_{datetime.now()}.png"
        #         cv2.imwrite(imagefile, output_image)
        #         print(f"{imagefile}, Succesfully Saved...!!")
        #         intensity_mean = process_image(output_image)
        #         intensity.append(intensity_mean)
        #         prev_res = res
        #     else:
        #         continue

        # if res == 9:
        #     # send command to turn off the led lamp
        #     end_time = time.time() - start_time
        #     print(f"elapsed time to acquisition: {end_time}s.")
        #     ser.write(b'1')
        #     responses = 0
        #     num = 0
        #     prev_res = 0    
        #     res_stop = False
        #     acquire = False
        #     res = 0
        #     res_ir = None
        #     kwargs_rest["status_gerak"] = False
        #     sampel = None
        #     # predict here
        #     result = detect(np.array(intensity[-8:]).reshape(1, -1))
        #     # send to api to display graph and image (there are 8 images for each fruit)
        #     print(f"Prediction Result: {result}")
        #     # send command to pneumatic_control
        #     # if result["result"] == 'MENTAH':
        #     #     # code is 2, 2 means for box 2 (int)
        #     #     send_signal_job(result)
        #         # todo: send command to move the conveyor

        #     capture = False
        #     # todo: after prediction is done, push to dashboard to display graph
        #     payload = {
        #         "prediction": result["result"],
        #         "label": ["LED1", "LED2", "LED3", "LED4", "LED5", "LED6", "LED7", "LED8"],
        #         "data": intensity
        #     }
        #     # res = requests.post(URL, json=payload)
        #     intensity = []
        #     time.sleep(2)

    st_device.acquisition_stop()
    st_datastream.stop_acquisition()


if __name__ == '__main__':
	fire.Fire()
 