#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import os
import torch
import time


path_model = os.path.join('model yolov4', 'yolov4-tiny_best.weights')
path_cfg = os.path.join('model yolov4', 'yolov4-tiny.cfg')


Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open(os.getcwd() + "/classes.txt", "r") as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv.dnn.readNet(path_model, path_cfg)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

print(model)

video_path = "datasets/MENTAH-DEPAN-7des22.avi"
cap = cv.VideoCapture(0)
starting_time = time.time()
frame_counter = 0
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if ret == False:
        break
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid], score)
        cv.rectangle(frame, box, color, 1)
        cv.putText(
           frame,
           label,
           (box[0], box[1] + 20),
           cv.FONT_HERSHEY_DUPLEX,
           0.7,
           color,
           2,
        )
    endingTime = time.time() - starting_time
    fps = round(frame_counter / endingTime, 2)
    # print(fps)
    cv.putText(
         frame, f"FPS: {fps}", (20, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1
        )
    cv.imshow("frame", frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()


# In[8]:


# %ls


# In[18]:


# !python yolov7\detect.py --weights best.pt --source video_sampel\F0_F1-DEPAN.mp4 --img 416 --conf-thres 0.55 --iou-thres 0.55


# In[15]:


# !pip install -r yolov7\requirements.txt


# In[ ]:




