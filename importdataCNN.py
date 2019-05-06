# Program that imports and preprocesses data to meet
# reqiuerments that CNN have on input data

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Define main function
def main():
    print("Loading datasets...")
    train = pd.read_csv("datasets/exoTrain.csv", encoding= "ISO-8859-1") #on data frame format
    test = pd.read_csv("datasets/exoTest.csv", encoding= "ISO-8859-1") #on data frame format

    # Converting the formate from dataframe to numpy arrays (matrices)
    # and defining x-values and y-values for both the test and training set
    x_train = np.array(train.drop('LABEL', axis=1)) #remove classification-column "label"
    x_test = np.array(test.drop('LABEL', axis=1))
    y_train = np.array(train.LABEL) #add classification-column "label"
    y_test = np.array(test.LABEL)
  
    # Plotting the unprocessed light curve
    plt.subplot(2, 1, 1)
    plt.plot(x_train[1, :], '.')
    plt.title('Unprocessed light curve')

    # Scale each observation to zero mean and unit variance
    x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / np.std(x_train, axis=1).reshape(-1,1))
    x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / np.std(x_test, axis=1).reshape(-1,1))

    # Preprocessing data
    x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=200)], axis=2)
    x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)

    # Plotting the processed light curve
    plt.subplot(2, 1, 2)
    plt.plot(x_train[1, :], '.')
    plt.title('Processed light curve')
    plt.show()

    # Construct the neural network
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=11, activation='relu', input_shape=x_train.shape[1:]))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(Flatten())
    model.add(Dropout(0.5)) #prevents overfitting
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25)) #prevents overfitting
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Define function that generates batch with equally positiv
    # and negative samples, and rotates them randomly in time
    def batch_generator(x_train, y_train, batch_size=32):
        half_batch = batch_size // 2
        x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32') #define empty batch for input
        y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32') #define empty batch for output

        yes_idx = np.where(y_train[:,0] == 2.)[0] 
        non_idx = np.where(y_train[:,0] == 1.)[0]

        while True:
            np.random.shuffle(yes_idx) #randomize indices
            np.random.shuffle(non_idx)
            x_batch[:half_batch] = x_train[yes_idx[:half_batch]] #half the batch has a positive label
            x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]] #the other half has a negative label
            y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
            y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]
            for i in range(batch_size):
                sz = np.random.randint(x_batch.shape[1])
                x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
     
            yield x_batch, y_batch
        

print("Before main")
if __name__ == '__main__':
    print("In main")
    main()