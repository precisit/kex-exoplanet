# Program that imports and preprocess our data to meet 
# requirments that CNN have on input data.

import pandas as pd
import numpy as np

from scipy.ndimage.filters import uniform_filter1d

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def main():
    trainSetPath = "datasets/exoTrain.csv"
    testSetPath = "datasets/exoTest.csv"
    print("Loading datasets...")
    df_train = pd.read_csv(trainSetPath, encoding= "ISO-8859-1")
    df_dev = pd.read_csv(testSetPath, encoding= "ISO-8859-1")
    
    print(df_train.info())

    # Converting the pandas dataframe to numpy arrays (matrices)
    df_train_x = df_train.drop('LABEL', axis=1)
    df_test_x = df_dev.drop('LABEL', axis=1)
    df_train_y = df_train.LABEL
    df_test_y = df_dev.LABEL
    X_train = np.array(df_train_x)
    Y_train = np.array(df_train_y)    
    X_test = np.array(df_test_x)
    Y_test = np.array(df_test_y)

    # Scale each observation to zeromean and unit variance

    X_train1 = ((X_train - np.mean(X_train, axis=1).reshape(-1,1)) / np.std(X_train, axis=1).reshape(-1,1))
    X_test1 = ((X_test - np.mean(X_test, axis=1).reshape(-1,1)) / np.std(X_test, axis=1).reshape(-1,1))

    #PLOTTING the first processed light curve
    plt.subplot(2, 1, 1)
    plt.plot(X_train[1, :], '.')
    plt.title('Unprocessed light curve')

    #Plotting the second processed light curve
    plt.subplot(2, 1, 2)
    plt.plot(X_train1[1, :], '.')
    plt.title('Processed light curve')
    plt.show()




print("Before main")
if __name__ == '__main__':
    print("In main")
    main()
