#Trying out code from @gabogarza

#Requires training and test data in a folder named datasets
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def np_X_Y_from_df(df):
    print("In np_x_y...")
    df_X = df.drop(['LABEL'], axis=1)
    X = np.array(df_X)
    Y_raw = np.array(df['LABEL']).reshape((len(df['LABEL']),1))
    Y = Y_raw == 2
    return X, Y

def main():
    trainSetPath = "datasets/exoTrain.csv"
    testSetPath = "datasets/exoTest.csv"
    print("Loading datasets...")
    df_train = pd.read_csv(trainSetPath, encoding = "ISO-8859-1")
    df_dev = pd.read_csv(testSetPath, encoding = "ISO-8859-1")

    print(df_train.info())

    # Generate X and Y dataframe sets
    df_train_x = df_train.drop('LABEL', axis=1)
    df_dev_x = df_dev.drop('LABEL', axis=1)
    df_train_y = df_train.LABEL
    df_dev_y = df_dev.LABEL
    X = np.array(df_train_x)
    Y = np.array(df_train_y)
    Xd= np.array(df_dev_x)
    Yd= np.array(df_dev_y)
    
    print(Y)

    model=LinearSVC()
    print("Training...")
    model.fit(X, Y)
    print("Finished training!")
    train_outputs=model.predict(X)
    test_outputs =model.predict(Xd)
    acc=accuracy_score(Y,train_outputs)
    recall_train = recall_score(Y, train_outputs)
    confM=confusion_matrix(Y,train_outputs)
    print(acc)
    print(recall_train)
    print("Confusion Matrix - Train Set")
    print(confM)

    confMd=confusion_matrix(Yd,test_outputs)
    print("Confusion Matrix - Test Set")
    print(confMd)

    #PLOTTING
    plt.plot(X[1, :], '.')
    plt.show()

print("Before main")
if __name__ == '__main__':
    print("In main")
    main()

