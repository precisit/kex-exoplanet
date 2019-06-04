# Detecting exoplanets using convolutional neural networks and support vector machine
This repository is connected to a bachelor thesis with the purpose of finding exoplanets with two machine learning algorithms. The bachelor thesis can be found in this repository as thesis.pdf. For this study two machine learning algorithms, support vector machine and convolutional neural networks, where developed and evaluated on a data set to determine which algorithm that performed the best. This project was implemented in pyhton, executing it in Google Colab on a GPU.

### Data set
The data set can be found at a host at [kaggle.com](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data?fbclid=IwAR04asE3i9QKo9SZru88wCxPsh-EIYYqUUN_8PMu1TdA_k0x5MM1dNg3OPg). It contains time series of light intensity of a star which binary classified in two catagories, "star with no exoplanet in orbit" or "star with an exoplanet in obrit". 
SVM drive data-folder:
https://drive.google.com/drive/folders/1QLQxg-BjKFzRfSUeOYc8y-AiiP-_GMip?usp=sharing

### Code
[CNN.py](https://github.com/precisit/kex-exoplanet/blob/master/CNN.py)
* Containing code for: 
  - Preprocessing of data
  - Training of the network
  - Validation of the network
  
[CNN_loadmodel.py](https://github.com/precisit/kex-exoplanet/blob/master/CNN_loadmodel.py) 
* Code for using an already trained model, saved on your computer

[graphics_preprocessing.py](https://github.com/precisit/kex-exoplanet/blob/master/graphics_preprocessing.py)
* Creates following graphics: 
  - Preprocessing steps for CNN
  - Raw data of light curves for the thesis
  
[SVM.py](https://github.com/precisit/kex-exoplanet/blob/master/SVM.py)

[SVM_gridsearch.ipynb](https://github.com/precisit/kex-exoplanet/blob/master/SVM_gridsearch.ipynb)

[SVM_nofeat_gridsearch.ipynb](https://github.com/precisit/kex-exoplanet/blob/master/SVM_nofeat_gridsearch.ipynb)

[SVM_preprocess.py](https://github.com/precisit/kex-exoplanet/blob/master/SVM_preprocess.py)

### Required packages
* Keras
* Pandas
* Numpy
* Matplotlib

### Code authors
Sofia Dreborg: [@sofiadreborg](https://github.com/sofiadreborg)

Maja Linderholm: [@majalinderholm](https://github.com/majalinderholm)

Jacob Tiensuu: [@JacobTiensuu](https://github.com/JacobTiensuu)

Fredrik Örn: [@fredrikorn](https://github.com/fredrikorn)
