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
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
from google.colab import drive
drive.mount('/content/gdrive')

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define main function
def main():
    print("Loading datasets...")
    train = pd.read_csv("gdrive/My Drive/datasets/exoTrain.csv", encoding= "ISO-8859-1") #on data frame format
    test = pd.read_csv("gdrive/My Drive/datasets/exoTest.csv", encoding= "ISO-8859-1") #on data frame format
    x_train = train.drop('LABEL', axis=1)
    x_test = test.drop('LABEL', axis=1)
    y_train = train.LABEL
    y_test = test.LABEL
    x_train = np.array(x_train)
    y_train = np.array(y_train).reshape((-1,1))-1
    x_test = np.array(x_test)
    y_test = np.array(y_test).reshape((-1,1))-1 
    
    print(y_train)
    print(y_train.shape)

    # Converting the formate from dataframe to numpy arrays (matrices)
    # and defining x-values and y-values for both the test and training set
    #raw_data = np.loadtxt("gdrive/My Drive/datasets/exoTrain.csv", skiprows=1, delimiter=',')
    #x_train = raw_data[:, 1:]
    #y_train = raw_data[:, 0, np.newaxis] - 1.
    #raw_data = np.loadtxt("gdrive/My Drive/datasets/exoTest.csv", skiprows=1, delimiter=',')
    #x_test = raw_data[:, 1:]
    #y_test = raw_data[:, 0, np.newaxis] - 1.
    #del raw_data
    
    #x_train = np.array(x_train)
    #y_train = np.array(y_train)
    #x_test = np.array(x_test)
    #y_test = np.array(y_test)
    
    #print(y_train)
    #print(y_train[1])
    #print(y_train.shape)
  
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
    
    #Define a function for shuffeling in unison
    def shuffle_in_unison(a, b):
      rng_state = np.random.get_state()
      np.random.shuffle(a)
      np.random.set_state(rng_state)
      np.random.shuffle(b)

    # Define function that generates batch with equally positive
    # and negative samples, and rotates them randomly in time
    def batch_generator(x_train, y_train, batch_size=32):
        half_batch = batch_size // 2
        x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32') #empty matrix for input
        y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32') #empty matrix for output

        # Find indicies for positive and negative labels
        while True:
            pos_idx = np.where(y_train[:,0] == 1)[0]
            neg_idx = np.where(y_train[:,0] == 0)[0]
            
            # Randomize the positive and negative indicies
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)

            # Let half of the batch have a positive classification and the other
            # half have a negative classification
            x_batch[:half_batch] = x_train[pos_idx[:half_batch]]
            x_batch[half_batch:] = x_train[neg_idx[half_batch:batch_size]] 
            y_batch[:half_batch] = y_train[pos_idx[:half_batch]]
            y_batch[half_batch:] = y_train[neg_idx[half_batch:batch_size]]
            
            # Shuffle batch
            shuffle_in_unison(x_batch,y_batch)

            # Generating new examples by rotating them in time
            for i in range(batch_size):
                sz = np.random.randint(x_batch.shape[1])
                x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
            yield x_batch, y_batch

    # # Define costume metrics
    # def recall(y_true,true_neg)
    #     return y_true//(y_true+true_neg)

    def precision(y_true, y_pred):
	    """Precision metric.
	
	    Only computes a batch-wise average of precision.
	
	    Computes the precision, a metric for multi-label classification of
	    how many selected items are relevant.
	    """
	    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	    precision = true_positives // (predicted_positives + K.epsilon())
	    return precision
	
	
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives // (possible_positives + K.epsilon())
        return recall

    # Compile model and train the model, make sure it converges
    model.compile(optimizer=Adam(1e-5), loss = 'binary_crossentropy', metrics=['accuracy'])
    hist = model.fit_generator(batch_generator(x_train, y_train, 32), \
                                validation_data=(x_test, y_test), \
                                verbose=0, epochs=5, \
                                steps_per_epoch=x_train.shape[0]//32)

    # Proceeding the training with faster learning rate
    model.compile(optimizer=Adam(4e-5), loss = 'binary_crossentropy', metrics=['accuracy',precision,recall])
    hist = model.fit_generator(batch_generator(x_train, y_train, 32), 
                                validation_data=(x_test, y_test), 
                                verbose=2, epochs=10,
                                steps_per_epoch=x_train.shape[0]//32)

    # Plot convergence rate
    #plt.plot(hist.history['recall'], color='g')
    #plt.plot(hist.history['val_recall'], color='r')
    #plt.title('Recall')
    #plt.show()
    #plt.plot(hist.history['precision'], color='g')
    #plt.plot(hist.history['val_precision'], color='r')
    #plt.title('Precision')
    #plt.show()
    plt.plot(hist.history['loss'], color='b')
    plt.plot(hist.history['val_loss'], color='r')
    plt.title('Loss')
    plt.show()
    plt.plot(hist.history['acc'], color='b')
    plt.plot(hist.history['val_acc'], color='r')
    plt.title('Accuracy')
    plt.show()

    # Make predictions for test data
    neg_idx = np.where(y_test == 0)[0]
    pos_idx = np.where(y_test == 1)[0]
    shuffle_in_unison(x_test,y_test)
    y_pred = model.predict(x_test)[:,0]
    
    pred = np.empty((1,len(y_pred)), dtype=object)
    pred = np.where(y_pred>=0.5, 1, 0)

    y_test = np.reshape(y_test,len(y_test))
    pred = np.reshape(pred,len(pred))
    print(y_test[0:5])
    print(pred[0:5])
    print(y_test.shape)
    print(pred.shape)
    
    # Create confusion matrix for training data
    y_test = pd.Series(y_test, name='Actual')
    pred = pd.Series(pred, name='Predicted')
    df_confusion = pd.crosstab(y_test, pred)
    print(df_confusion)    
    matrix = confusion_matrix(pred, y_test)
    print(matrix)  

        
print("Before main")
if __name__ == '__main__':
    print("In main")
    main()