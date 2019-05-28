
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage, fft, integrate
from scipy.ndimage.filters import uniform_filter1d
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score,\
                recall_score, confusion_matrix, fbeta_score, auc,\
                  precision_recall_curve, average_precision_score
from inspect import signature
from preprocess import LightFluxProcessor, StandardScaler

def main():
    trainSetPath = "datasets/exoTrain.csv"  # Loads datasets, requires a folder named "datasets"
    testSetPath = "datasets/exoTest.csv"    # containing the data files in your current folder
    print("Loading datasets...")
    df_train = pd.read_csv(trainSetPath, encoding = "ISO-8859-1")
    df_test = pd.read_csv(testSetPath, encoding = "ISO-8859-1")

    # Generate X and Y dataframe set
    df_train_x = df_train.drop('LABEL', axis=1) 
    df_test_x = df_test.drop('LABEL', axis=1)
    df_train_y = df_train.LABEL
    df_test_y = df_test.LABEL
    
    X_train = np.array(df_train_x)    #
    Y_train = np.array(df_train_y)    # The raw input/output data for
    X_test= np.array(df_test_x)     # both train and test sets as np.arrays
    Y_test= np.array(df_test_y)     #

    #Adding mirrored series
    extra = np.flip(X_train[0:37,:], axis=-1)
    extraY = Y_train[0:37]
    X_train = np.append(X_train,extra, axis=0)
    Y_train = np.append(Y_train,extraY,axis=0) #Kan vara bra att lägga till fler exempel för test-setet också
    dextra = np.flip(X_test[0:5,:], axis=-1)
    dextraY = Y_test[0:5]
    X_test = np.append(X_test, dextra, axis=0)
    Y_test = np.append(Y_test,dextraY,axis=0)

    Y_train=Y_train-1       #
    Y_test=Y_test-1     # To get postives to 1 and negatives to 0

    # Process dataset - choose which should be used
    LFP = LightFluxProcessor(
        fourier=True,
        normalize=False,             
        gaussian=False,             
        standardize=False)      
    X_train, X_test = LFP.process(X_train, X_test)

    #Normalization - gives better recall but worse precision
    # X_train = ((X_train - np.mean(X_train, axis=1).reshape(-1,1)) / np.std(X_train, axis=1).reshape(-1,1))
    # X_test = ((X_test - np.mean(X_test, axis=1).reshape(-1,1)) / np.std(X_test, axis=1).reshape(-1,1))


    #TRAINING AND EVALUATING THE SVC
    c_w = {0: 1,  #Directory for trying out different class weights
          1: 1}      #Change to kernel=c_w to use

    model=SVC(kernel='linear', gamma='scale', class_weight='balanced', probability=True, max_iter=10000)   # Choosing model
    print("Training...")
    model.fit(X_train, Y_train)            # Choose which data to train on
    print("Finished training!")
    print('')
    print("Making predictions..")
    train_outputs=model.predict(X_train)     # Making predictions
    test_outputs =model.predict(X_test)    #

    train_prob = model.predict_proba(X_train)[:,1]
    test_prob = model.predict_proba(X_test)[:,1]
    print("Finished predictions!")

    #region Various metrics for performance
    ap_train = average_precision_score(Y_train, train_prob)
    ap_test = average_precision_score(Y_test, test_prob)

    #Precision recall curve
    precision, recall, thresholds = precision_recall_curve(Y_train,train_prob) #train
    precision_d, recall_d, thresholds_d = precision_recall_curve(Y_test,test_prob) #test
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                if 'step' in signature(plt.fill_between).parameters
                else {})
    plt.step(recall, precision, color='b', alpha=0.2,
            where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('SVM: Precision-Recall Curve')
    plt.show()

    accuracy_train=accuracy_score(Y_train,train_outputs)
    accuracy_test=accuracy_score(Y_test,test_outputs)
    precision_train=precision_score(Y_train,train_outputs)
    precision_test=precision_score(Y_train,test_outputs)
    recall_train = recall_score(Y_train, train_outputs)
    recall_test = recall_score(Y_test,test_outputs)
    f1_train = f1_score(Y_train,train_outputs)
    f1_test = f1_score(Y_test,test_outputs)

    print("AUC training set: %.3f" %ap_train )
    print("AUC test set: %.3f" %ap_test )
    print("Accuracy training set: %.3f" %accuracy_train)
    print("Accuracy test set: %.3f" %accuracy_test)
    print("Precision training set: %.3f" %precision_train)
    print("Precision test set: %.3f" %precision_test)
    print("Recall training set: %.3f" %recall_train)
    print("Recall test set: %.3f" %recall_test)
    print("F1 score training set: %.3f" %f1_train)
    print("F1 score test set: %.3f" %f1_test)
    print(' ')
    confM=confusion_matrix(Y_train,train_outputs)
    print("Confusion Matrix - Train Set")
    print(confM)

    confMd=confusion_matrix(Y_test,test_outputs)
    print("Confusion Matrix - Test Set")
    print(confMd) 
    #endregion

    input()

if __name__ == '__main__':
    main()