from tkinter import NORMAL
# this will be used to process movement detection

status_multi = {
    "no_work": "Multi Spectral System is off",
    "start": "Multi Spectral System is starting",
    "no_object": "System is on but no object inside",
    "move_det": "System is on and object inside",
    "get_data": "System is acquiring data",
    "stop": "Data acquisition is finished"
}

NUM_DATA = 3
state_service = {
    "multi_panel": NORMAL,
    "volume": NORMAL,
    "all": NORMAL
}

status_service = {
    "multi": True,
    "volume": False,
    "all": False,
}

if status_service["multi"]:
    mvDet = True
    status_capturing = True
    status_gerak = True
    multiCam = True
    volDet = False
else:
    mvDet = False
    status_capturing = False
    status_gerak = False
    multiCam = False
    volDet = True


# if status_service["volume"]:
    
# else:
#     volDet = True

PORT_ARDUINO = 'COM3'
PORT_FW = "COM6"

status = True
ledCount = 0
FolderName = ""
mean = None
first_frame = None
responses = {}

args_mvDet = {
    "mean": None,
    "first_frame": None,
}

kwargs_rest = {
    "status_capturing": False,
    "status_gerak": False,
    "ledCount": 0,
    "init_capturing": False,
    "responses": {},
    "FolderName": None,
    "category": None,
    "part": None,
    "key_input": None,
}

args_vol = {}

set_vol = {
    "bluelow": 150,
    "greenlow": 150,
    "redlow": 240,
    "blueup": 255,
    "greenup": 255,
    "redup": 255,
    "result": 0,
    "minor_min": 0,
    "minor_max": 0,
    "major_min": 0,
    "major_max": 0,
    "minor_length": 0,
    "major_length": 0,
}
op = 14.020046
pi = 22/7

