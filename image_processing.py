import pandas as pd 
import numpy as np 
import cv2
import glob
import sys, os
import fire

def test_process():
	black_ref = cv2.imread('image_reference/blk.png')
	print(black_ref.shape)
	# black_ref = black_ref.reshape((614, 326, 1))
	black_ref = black_ref[:, :, 0]
	white_ref = cv2.imread('image_reference/wr.png')
	# white_ref = white_ref.reshape((614, 326, 1))
	white_ref = white_ref[:, :, 0]

	print(black_ref.shape, white_ref.shape, ">>>")	


def process_image(frame_multi):
	frame_multi = frame_multi.reshape(614, 326, 1)
	black_ref = cv2.imread('image_reference/blk.png')[:, :, 0]
	black_ref = black_ref.reshape(614, 326, 1)
	white_ref = cv2.imread('image_reference/wr.png')[:, :, 0]
	white_ref = white_ref.reshape(614, 326, 1)

	Y = np.subtract(white_ref, black_ref)
	h1,w1,l1 = Y.shape
	for i in range(l1):
	    y = Y[:, :, i]
	    m, n = y.shape
	    for s in range(m):
	        for t in range(n):
	            if y[s][t] < 0:
	                y[s][t] = 0
	            if y[s][t] == 0:
	                y[s][t] = 1


	X = np.subtract(frame_multi, black_ref)
	h1,w1,l1 = X.shape
	for i in range(l1):
	    temp = X[:, :, i]
	    m, n = temp.shape
	    for s in range(m):
	        for t in range(n):
	            if temp[s][t] < 0:
	                temp[s][t] = 0
	result_array = np.divide(X, Y)

	# roi = [178, 97, 128, 105]
	# roi = [222, 77, 129, 136]
	roi = [255, 116, 114, 89]
	res = result_array[(roi[1]):(roi[1]+roi[3]), (roi[0]):(roi[0]+roi[2])]
	roi_mean = np.mean(res)
	return roi_mean

if __name__ == '__main__':
	fire.Fire()
