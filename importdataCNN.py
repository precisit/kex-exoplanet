# PROGRAM THAT IMPORTS AND PREPROCESSES DATA TO MEET
# REQUIRMENTS THAT CNN HAVE ON INPUT DATA

# IMPORT PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# DEFINE MAIN-FUNCTION
def main():
    trainSetPath = "datasets/exoTrain.csv" #define path for data files
    testSetPath = "datasets/exoTest.csv"
    print("Loading datasets...")
    train = pd.read_csv(trainSetPath, encoding= "ISO-8859-1") #on data frame format
    test = pd.read_csv(testSetPath, encoding= "ISO-8859-1") #on data frame format
    print(train.info())

    # CONVERTING THE PANDAS DATAFRAM TO NUMPY ARRAYS (MATRICES)
    train_x = train.drop('LABEL', axis=1) #removes "label" from x-values
    test_x = test.drop('LABEL', axis=1) 
    train_y = train.LABEL #picks "label" to y-values
    test_y = test.LABEL
    x_train = np.array(train_x) #changes format form data frame to numpy array
    y_train = np.array(train_y)
    x_test = np.array(test_x) 
    y_test = np.array(test_y)

    # SCALE EACH OBSERVATION TO ZERO MEAN AND UNIT VARIANCE
    x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / np.std(x_train, axis=1).reshape(-1,1))
    x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / np.std(x_test, axis=1).reshape(-1,1))

    # PLOTTING THE FIRST PROCESSED LIGHT CURVE
    plt.subplot(2, 1, 1)
    plt.plot(x_train[1, :], '.')
    plt.title('Unprocessed light curve')

    # PLOTTING THE SECOND PROCESSED LIGHT CURVE
    plt.subplot(2, 1, 2)
    plt.plot(x_test[1, :], '.')
    plt.title('Processed light curve')
    plt.show()

print("Before main")
if __name__ == '__main__':
    print("In main")
    main()
