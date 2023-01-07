import serial
import fire
import time
import sys, os
from config import kwargs_rest
import io,re
import cv2
import stapipy as st
import numpy as np
from auto_functions_opencv import CMyCallback
sys.path.append(os.getcwd())
# from image_processing import process_image
# from predict import detect
# from pneumatic_control import send_signal_job

# initialize for arduino + camera
ser = serial.Serial(port='COM6', baudrate=9600)
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

FolderName = "DataMulti - PTPN5/PTPN5 - WhiteRef"
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
    start_time = None

    fraksi_dic = {'1': 'MENTAH', '2': 'MATANG'}
    skenario_dic = {'1': 'D', '2': 'B'}
    responses = {}    # sampel = 0

    sampel = input('masukkan nama sampel: ')
    print(f'jenis sampel = {fraksi_dic}')
    jenis = input('masukkan jenis: ')
    print(f'skenario = {skenario_dic}')
    skenario_acq = input('masukkan skenario: ')
    fraksi = fraksi_dic[str(jenis)]
    skenario = skenario_dic[str(skenario_acq)]
    intensity = []

    while run:
        kwargs_rest['key_input'] = cv2.waitKey(1)
        if sampel:
            kwargs_rest['responses']['moved'] = '1'
        else:
            kwargs_rest['responses']['moved'] = '0'

        if kwargs_rest['key_input'] == ord('q'):
            break 

        output_image = my_callback.image
        # output_image = cv2.resize(output_image, (1024, 540))
        if output_image is not None:
            output_image = output_image

        if kwargs_rest["responses"]['moved'] == '1':
            if kwargs_rest["init_capturing"] == False:
                kwargs_rest["status_gerak"] = True
                kwargs_rest["init_capturing"] = True
                kwargs_rest["status_capturing"] = True
                kwargs_rest["init_capturing"] = True

            # check the folder created
            if not os.path.exists(FolderName):
                os.makedirs(FolderName)
            
        if kwargs_rest["status_gerak"] == True:
            num += 1
        # else:
        #     print("There is Palm Fruit detected. waiting for 5 seconds.")
        #     time.sleep(5)


        # res_ir = ser_ir.readline().decode()
        # print(f"Response from proximity: {res_ir}")
        # if res_ir[:6] == "BUAH":
        #     print("Found a object")
        #     time.sleep(6)

        #     if (angka == 5 and kwargs_rest["status_gerak"] == True):

        if num > 1 and kwargs_rest["status_gerak"] == True:
            start_time = time.time()
            ser.write(str(num).encode())
            response = ser.readline().decode()
            if "ready" in response and start_read is None:
                start_read = True            

        if start_read:
            try:
                res = int(response)
            except:
                continue

            cv2.imwrite(f"{FolderName}/{fraksi}_S{sampel}_LED{res}_{skenario}.png", output_image)
            print(f"{FolderName}/{fraksi}S{sampel}_LED{res}_{skenario}.png, Succesfully Saved!!")
            # intensity_mean = process_image(output_image)
            # intensity.append(intensity_mean)

            if res == 9:
                # send to turn off the led lamp
                ser.write(str(1).encode())

        output_image = my_callback.image
        if output_image is not None:
            cv2.imshow('frame', output_image)

        if num == 9:
            num = 0
            kwargs_rest["status_gerak"] = False

        if res == 9:
            # result = detect(np.array(intensity[-8:]).reshape(1, -1))
            # # send to api to display graph and image (there are 8 images for each fruit)
            # print(f"Prediction Result: {result}")
            # if result["result"] == 'MENTAH':
            #     # code is 2, 2 means for box 2 (int)
            #     send_signal_job(result)
            #     # todo: send command to move the conveyor
            # end_time = time.time() - start_time
            # print(f"elapsed time to acquisition: {end_time}s.")
            responses = 0
            num = 0
            res = 0
            start_read = None
            kwargs_rest["status_capturing"] = False
            kwargs_rest["status_gerak"] = False
            kwargs_rest["init_capturing"] = False
            kwargs_rest["responses"]['moved'] = None
            sampel = None
            # stop image acquisition
            run = False
            print("Image acquisition is done.")
            # return turnOn
            # run = True
            # time.sleep(1)
            # print("Waiting 1s to stop.")

    st_device.acquisition_stop()           
    st_datastream.stop_acquisition()


if __name__ == '__main__':
	fire.Fire()
 