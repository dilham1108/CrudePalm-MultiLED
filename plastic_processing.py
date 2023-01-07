# ---------TOLONG JANGAN DI EDIT---------- #
# ---------FILE INI DIGUNAKAN SEBAGAI ARSIP----#


import cv2
import numpy as np
import pandas as pd
import glob
import sys
import os

file_dir = 'DataMulti - PTPN5/PTPN5 - 19 Des 22'
# wr_dir = 'DataSet-Dilham/WR/'
save_dir = 'DataSet-Dilham/PP-Processed'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# black_ref = cv2.imread('BlackReference.png')
# black_ref = cv2.resize(black_ref, (648, 486))
# wr_roi = [154, 82, 307, 163]
img_roi = [154, 82, 307, 163]

# list_wr = glob.glob(wr_dir+'*')
# weref = {'f1': list_wr[0], 'f2' : list_wr[2], 'f3' : list_wr[3], 'f4' : list_wr[4], 'f5' : list_wr[5],
# 		'f6' : list_wr[6], 'f7' : list_wr[7], 'f8' : list_wr[8], 'f9' : list_wr[9], 'f10' : list_wr[1]} 

dataframe = pd.DataFrame()

for s in range(1, 1000):
	intensity = {'sampel': [], 'skenario' : [],
					'LED2': [], 'LED3': [], 'LED4': [], 'LED5': [], 'LED6': [],
					'LED7': [], 'LED8': [], 'LED9': []}
	sample = glob.glob(f'{file_dir}/*{s}_*')
	# print(s)
	for s in sample:
		print(s)
	depan = []
	belakang = []

	for i in sample:
		skenario = i.split('_')[3].split('.')[0]
		# print(skenario)
		if(skenario == 'D'):
			depan.append(i)
		if(skenario == 'B'):
			belakang.append(i)
	# sys.exit()
	
	for i in depan:
		intensity['sampel'] = [s]
		skenario = i.split('_')[3].split('.')[0]
		intensity['skenario'] = [skenario]
		nfilter = i.split('_')[2] 
		#'DataSet-Dilham/Dilham-12Agustus2022-PP\\f9-1_p3_(PP)_.png'
		#Dilham-Multispektral/Dilham-8Agustus2022-PET\\f1-31_p1_(PET)_T-BKS.png
		# wr = cv2.imread(weref[nfilter])
		# wr = cv2.resize(wr[wr_roi[1]:wr_roi[1]+wr_roi[3], wr_roi[0]:wr_roi[0] + wr_roi[2]], (648, 486))
		citra = cv2.imread(i)
		# citra = cv2.resize(citra, (648, 486))
		res = np.array(citra)
		# y = np.add(wr, black_ref)
		# res = np.divide(x, y)
		# filename = i.split('\\')[1]
		# cv2.imwrite(f'{save_dir}/{filename}', res)
		# print(f'{filename} image successfully saved')
		res = res[img_roi[1]:img_roi[1]+img_roi[3], img_roi[0]:img_roi[0] + img_roi[2]]
		mean = np.mean(res)
		intensity[nfilter] = mean

	df = pd.DataFrame(intensity, index=[1])
	dataframe = dataframe.append(df)
	depan = []
	print(f'intensity df >>>>>>>>> \n {df}')
	print(f'dataframe >>>>>>>>> \n {dataframe}')
	intensity = {'sampel': [], 'skenario' : [],
					'LED2': [], 'LED3': [], 'LED4': [], 'LED5': [], 'LED6': [],
					'LED7': [], 'LED8': [], 'LED9': []}

	for i in belakang:
		intensity['sampel'] = [s]
		# posisi = i.split('_')[1]
		# intensity['citra'] = [posisi]
		skenario = i.split('_')[3].split('.')[0]
		intensity['skenario'] = [skenario]
		nfilter = i.split('_')[2] #Dilham-Multispektral/Dilham-8Agustus2022-PET\\f1-31_p1_(PET)_T-BKS.png
		# wr = cv2.imread(weref[nfilter])
		# wr = cv2.resize(wr[wr_roi[1]:wr_roi[1]+wr_roi[3], wr_roi[0]:wr_roi[0] + wr_roi[2]], (648, 486))
		citra = cv2.imread(i)
		# citra = cv2.resize(citra, (648, 486))
		res = np.array(citra)
		# y = np.add(wr, black_ref)
		# res = np.divide(x, y)
		# filename = i.split('\\')[1]
		# cv2.imwrite(f'{save_dir}/{filename}', res)
		# print(f'{filename} image successfully saved')
		res = res[img_roi[1]:img_roi[1]+img_roi[3], img_roi[0]:img_roi[0] + img_roi[2]]
		mean = np.mean(res)
		intensity[nfilter] = mean

	df = pd.DataFrame(intensity, index=[1])
	dataframe = dataframe.append(df)
	belakang = []
	print(f'intensity df >>>>>>>>> \n {df}')
	print(f'dataframe >>>>>>>>> \n {dataframe}')
	intensity = {'sampel': [], 'skenario' : [],
					'LED2': [], 'LED3': [], 'LED4': [], 'LED5': [], 'LED6': [],
					'LED7': [], 'LED8': [], 'LED9': []}



# dataframe.to_excel('Intensity-PP.xlsx')
