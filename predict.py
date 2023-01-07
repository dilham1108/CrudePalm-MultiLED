""" this script will be used to detect the new data
with latest model
"""
import numpy as np
import sys, os
import pickle
import fire
from sklearn.preprocessing import StandardScaler as sc
from tensorflow.keras.models import load_model

sys.path.append(os.getcwd())
from training import dataset_testing


def load_sc(picklefile):
	""" Load Standard Scaler params"""

	infile = open(picklefile, 'rb') 
	sc = pickle.load(infile)
	return sc


def load_model_multi(modelfile):
	""" load multi_mode.h5 file """

	model = load_model(modelfile)
	return model


def convert_to_text(result):
	""" Convert result to text """

	list_res = {
		0: "MENTAH",
		1: "MATANG",
		2: 'LEWAT MATANG'
	}
	result_txt = {}

	if result[0] in list_res.keys():
		result_txt["result"] = list_res[result[0]]
	return result_txt


def single_predict(data, modelfile, sc):
	""" 
	single prediction.
	:data: should in array, e.g
	[[0.50029288 0.64601646 0.65341135]]
	"""

	model = load_model_multi(modelfile)
	# transform data
	data = sc.transform(data)
	# predict the data
	result = model.predict(data)

	# Converting predictions to label
	y_pred_result = list()
	for i in range(len(result)):
	    y_pred_result.append(np.argmax(result[i]))

	return y_pred_result


def detect(data=[], modelfile="models/multi_model.h5",
				picklefile="models/StandardScalerValue"):
	"""
	Detect new data.
	:data: should in array (1,3)
	"""
	sc = load_sc(picklefile)
	result = single_predict(data, modelfile, sc)
	result_txt = convert_to_text(result)
	print(result_txt)
	return result_txt


def test_predict(filename="datasets/data_testing_8fil_backup.xlsx", 
				modelfile="models/multi_model.h5",
				picklefile="models/StandardScalerValue"):
	""" Test predicition """

	x_test, y_test = dataset_testing(filename)
	x_test = x_test[0].reshape(1,3)
	sc = load_sc(picklefile)
	result = single_predict(x_test, modelfile, sc)
	result_txt = convert_to_text(result)
	return result_txt


# ------------------ this is the new version from ilkoms' students
def jst_detect(data=[], modelfile="models/multi_jst.h5"):
	"""
	Detect new data.
	:data: should in array (1,3)
	"""
	result = jst_predict(data, modelfile)
	result_txt = convert_to_text(result)
	print(result_txt)
	return result_txt

def jst_predict(data, modelfile, sc):
	""" 
	single prediction.
	:data: should in array, e.g
	[[0.50029288 0.64601646 0.65341135]]
	"""
	model = load_model_multi(modelfile)
	# predict the data
	result = model.predict(data)

	# Converting predictions to label
	y_pred_result = list()
	for i in range(len(result)):
	    y_pred_result.append(np.argmax(result[i]))

	return y_pred_result

if __name__ == '__main__':
	fire.Fire()




