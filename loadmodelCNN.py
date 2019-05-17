# Program that imports and preprocesses data to meet
# reqiuerments that CNN have on input data

# Import packages
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from keras import backend as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
from google.colab import drive
drive.mount('/content/gdrive')
import h5py

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    loaded_model.compile(optimizer=Adam(1e-5), loss = 'binary_crossentropy', metrics=['accuracy'])
    loaded_model.summary()

    # Converting the formate from dataframe to numpy arrays (matrices)
    # and defining x-values and y-values for both the test and training set
    print("Loading datasets...")
    test = pd.read_csv("gdrive/My Drive/datasets/exoTest.csv", encoding= "ISO-8859-1") #on data frame format
    x_test = test.drop('LABEL', axis=1)
    y_test = test.LABEL
    x_test = np.array(x_test)
    y_test = np.array(y_test).reshape((-1,1))-1 

    # Scale each observation to zero mean and unit variance
    x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / np.std(x_test, axis=1).reshape(-1,1))

    # Preprocessing data
    x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)


    # evaluate loaded model on test data
    loaded_model.compile(optimizer=Adam(1e-5), loss = 'binary_crossentropy', metrics=['accuracy'])

    y_prediction = loaded_model.predict(x_test)[:,0]
    predction = np.empty((1,len(y_prediction)), dtype=object)
    prediction = np.where(y_prediction>=0.5,1,0)
    print(prediction)

    y_test = np.reshape(y_test,len(y_test))
    prediction = np.reshape(prediction,len(prediction))
        
    # Create confusion matrix for training data
    y_test = pd.Series(y_test, name='Actual')
    prediction = pd.Series(prediction, name='Predicted')
    conf_matrix = pd.crosstab(y_test, prediction)
    print(conf_matrix)
    print(conf_matrix.loc[0,0])
        
    # Define values in confusion matrix
    true_pos = conf_matrix.loc[1,1]
    false_pos = conf_matrix.loc[0,1]
    true_neg = conf_matrix.loc[0,0]
    false_neg = conf_matrix.loc[1,0]
        
    # Calculate precision and recall
    precision = true_pos // (true_pos + false_pos)
    print(precision)
    recall = true_pos // (true_pos + false_neg)
    print(recall)

print("Before main")
if __name__ == '__main__':
    print("In main")
    main()