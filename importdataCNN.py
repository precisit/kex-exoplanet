# Program that imports and preprocesses data to meet
# reqiuerments that CNN have on input data

# Import packages
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

    # Define function that generates batch with equally positive
    # and negative samples, and rotates them randomly in time
    def batch_generator(x_train, y_train, batch_size=32):
        half_batch = batch_size // 2
        x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32') #empty matrix for input
        y_batch = np.empty((batch_size), dtype='float32') #empty matrix for output

        # Find indicies for positive and negative labels
        while True:
            pos_idx = np.where(y_train == 2)[0]
            neg_idx = np.where(y_train == 1)[0]
            
            # Randomize the positive and negative indicies
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)

            # Let half of the batch have a positive classification and the other
            # half have a negative classification
            x_batch[:half_batch] = x_train[pos_idx[:half_batch]] 
            x_batch[half_batch:] = x_train[neg_idx[half_batch:batch_size]] 
            y_batch[:half_batch] = y_train[pos_idx[:half_batch]]
            y_batch[half_batch:] = y_train[neg_idx[half_batch:batch_size]]

            # Generating new examples by rotating them in time
            for i in range(batch_size):
                sz = np.random.randint(x_batch.shape[1])
                x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
            yield x_batch, y_batch

    # Compile model and train the model, make sure it converges
    model.compile(optimizer=Adam(1e-5), loss = 'binary_crossentropy', metrics=['accuracy'])
    hist = model.fit_generator(batch_generator(x_train, y_train, 32), \
                                validation_data=(x_test, y_test), \
                                verbose=0, epochs=5, \
                                steps_per_epoch=x_train.shape[0]//32)

    # Proceeding the training with faster learning rate
    model.compile(optimizer=Adam(4e-5), loss = 'binary_crossentropy', metrics=['accuracy'])
    hist = model.fit_generator(batch_generator(x_train, y_train, 32), 
                                validation_data=(x_test, y_test), 
                                verbose=2, epochs=40,
                                steps_per_epoch=x_train.shape[0]//32)

    # Plot convergence rate
    plt.plot(hist.history['loss'], color='b')
    plt.plot(hist.history['val_loss'], color='r')
    plt.show()
    plt.plot(hist.history['acc'], color='b')
    plt.plot(hist.history['val_acc'], color='r')
    plt.show()

    # Make predictions for test data
    neg_idx = np.where(y_test[:,0] == 1.)[0]
    pos_idx = np.where(y_test[:,0] == 2.)[0]
    y_hat = model.predict(x_test)[:,0]
        
print("Before main")
if __name__ == '__main__':
    print("In main")
    main()