import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import sys
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from kerastuner.tuners import RandomSearch
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model



def dataset_training(filename):
    datasets = pd.read_excel(filename, sheet_name="Sheet1")
    x_train = datasets.iloc[:, 1:].values
    y_train = datasets.iloc[:, 0:1].values  
    return x_train, y_train

def dataset_testing(filename):
    datasets = pd.read_excel(filename, sheet_name="Sheet1")
    x_test = datasets.iloc[:, 1:].values
    y_test = datasets.iloc[:, 0:1].values   
    return x_test, y_test


def normalize_data(x_train):
    """
    Normalizing the data.
    x_train: dataframe
    """
    sc = StandardScaler()
    x_normalized = sc.fit_transform(x_train)

    # store sc value in pickle format
    picklefile = "models/StandardScalerValue"
    outfile = open(picklefile, 'wb')
    pickle.dump(sc, outfile)
    outfile.close()

    return sc, x_normalized


def transform_data(sc, x_train):
    """ transform data """

    x_normalized = sc.transform(x_train)
    return x_normalized


def one_hot_encode(y_train):
    """ One hot encode. convert label to vector """
    ohe = OneHotEncoder()
    y_encode = ohe.fit_transform(y_train).toarray()
    return y_encode


def split_data_train(x_train, y_train):
    """ Train test split of model """
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size = 0.1, random_state = 0
    )
    return x_train, x_test, y_train, y_test


def bool_result_multi(y_pred_result, y_test, x_test, y_test_encoded):
    columns = ["Label", "LED1", "LED2", "LED3", "LED4", "LED5", "LED6", "LED7", "LED8", "Result"]
    result = []

    length = len(y_pred_result)
    for idx in range(length):
        if y_pred_result[idx] == y_test_encoded[idx]:
            result.append([True])
        else:
            result.append([False])
    result = np.array(result)

    all_data = np.hstack((y_test, x_test, result))
    all_data = pd.DataFrame(all_data, columns=columns)

    return all_data


def init_ann_model():
    """ initialization for ann model """
    model = Sequential()
    input_dim = 8
    output_dim = 2

    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    # model.add(Dropout(0.9))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.9))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.9))
    model.add(Dense(output_dim, activation='softmax'))
    return model


if __name__ == '__main__':

    TEST = True
    MITA = True
    BOOL_RESULT = True
    filename_training = "datasets/data_train_8fil_backup.xlsx"
    filename_testing = "datasets/data_testing_8fil_backup.xlsx"
    filename_training_out = "models/error_accuracy_training_3fil.xlsx"
    outfile = "models/comparison_result_3fil.csv"

    x_train, y_train = dataset_training(filename_training)
    sc, x_train_norm = normalize_data(x_train)
    y_train_encoded = one_hot_encode(y_train)
    x_train = x_train_norm
    y_train = y_train_encoded

    data = {
        "error": [],
        "accuracy": [],
    }

    
    # initialization for ann model
    model = init_ann_model()

    #To visualize neural network
    model.summary()

    
    if TEST:
        # this is for data testing()
        x_test, y_test = dataset_testing(filename_testing)
        x_test = transform_data(sc, x_test)
        y_test_origin = y_test
        y_test = one_hot_encode(y_test)
    else:
        _, y_test_file = dataset_testing(filename_testing)
        y_test_origin = y_test_file
        # split the training data to training data and testint data
        _, x_test, _, y_test = split_data_train(x_train, y_train)


    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train,
                        y_train,
                        epochs=100,
                        verbose=1,
                        batch_size=64)

    model.save('models/multi_model.h5')
    _, acc_train = model.evaluate(x_train, y_train, verbose=0)
    
    print(f"> Accuracy is {acc_train * 100} %  ")
    err = history.history["loss"]
    accuration = history.history["accuracy"]

    # # ---------- get result training --------------
    data["error"] = err
    data["accuracy"] = accuration
    df = pd.DataFrame(data)
    df.to_excel(filename_training_out)

    y_pred = model.predict(x_test)

    # Converting predictions to label
    y_pred_result = list()
    for i in range(len(y_pred)):
        y_pred_result.append(np.argmax(y_pred[i]))

    # Converting one hot encoded test label to label
    y_test_encoded = list()

    for i in range(len(y_test)):
        y_test_encoded.append(np.argmax(y_test[i]))

    if BOOL_RESULT:
        df = bool_result_multi(y_pred_result, y_test_origin, x_test, y_test_encoded)
        
        df["testing_label1"] = list(y_test[:, 0])
        df["testing_label2"]= list(y_test[:, 1])
        if not MITA:
            df["testing_label3"]= list(y_test[:, 2])

        df["original_prediction1"] = list(y_pred[:, 0])
        df["original_prediction2"] = list(y_pred[:, 1])

        if not MITA:
            df["original_prediction3"] = list(y_pred[:, 2])

        df.to_csv(outfile)
        print(f">> the file generated: {outfile}")

    acc_value = accuracy_score(y_pred_result, y_test_encoded)
    print('>> Accuracy value for testing is:', acc_value * 100)
    

    fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2)
    fig.suptitle('Metrics for training and testing analysis', y=1)
    fig.tight_layout(pad=3.0)

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['accuracy'])
    ax1.set_title('Accuracy Model \n for Training Data ', fontweight="bold", fontsize=6)
    ax1.set_ylabel('Accuracy', fontsize = 8.0)
    ax1.set_xlabel('Epoch', fontsize = 6.0)
    text = {"test_acc": round(acc_value * 100, 1),
            "train_acc": round(acc_train * 100, 1)}
    ax1.text(0.95, 0.01, f"test acc: {text['test_acc']}% | train_acc: {text['train_acc']}",
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color='green', fontsize=4, fontweight="bold")
    ax1.legend(['loss', 'accurayc'], loc='upper left', fontsize = 5.0)


    # Using testing data as validation data.
    history1 = model.fit(x_train, y_train, 
        validation_data = (x_test, y_test), 
        epochs=100, 
        batch_size=64
    )

    ax2.plot(history1.history['accuracy'])
    ax2.plot(history1.history['val_accuracy'])
    ax2.set_title('Accuracy Model \n for Training and Testing Data', fontweight="bold", fontsize=6)
    ax2.set_ylabel('Accuracy', fontsize = 8.0)
    ax2.set_xlabel('Epoch', fontsize = 6.0)
    ax2.legend(['Train', 'Test'], loc='upper left', fontsize = 5.0)

    ax3.plot(history1.history['loss'])
    ax3.plot(history1.history['val_loss'])
    ax3.set_title('Loss Model \n for Training and Testing Data', fontweight="bold", fontsize=6)
    ax3.set_ylabel('Loss', fontsize = 8.0)
    ax3.set_xlabel('Epoch', fontsize = 6.0)
    ax3.legend(['Train', 'Test' ], loc='upper left', fontsize = 5.0)

    plt.show()