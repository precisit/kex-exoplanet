# Program that imports and preprocess our data to meet 
# requirments that CNN have on input data.

import pandas as pd
import numpy as np

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
    df_dev_x = df_dev.drop('LABEL', axis=1)
    df_train_y = df_train.LABEL
    df_dev_y = df_dev.LABEL
    X = np.array(df_train_x)
    Y = np.array(df_train_y)    
    Xd = np.array(df_dev_x)
    Yd = np.array(df_dev_y)

    

print("Before main")
if __name__ == '__main__':
    print("In main")
    main()