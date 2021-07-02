# Bitcoin Price Prediction | CS542 - Machine Learning

This is the implementation by group 4 of the final project for CS 542 - Machine Learning.

In this project, we are applying various machine learning techniques in order to predict Bitcoin prices and then comparing each model to a set of baseline strategies to see how each model performs. The goal of this project is to construct a trading strategy that is based around a machine learning technique that is capable of beating more simplistic strategies such as buying and holding.
Our model will be divided into four categories, that is: Long Short Term Memory Recurrent Neural Net, Simple Recurrent Neural Net, Linear Regression, and Random Walk model.

## Models

Our code is divided into 4 models:
- Long Short Term Memory Recurrent Neural Net (LSTM-RNN)
- Simple Recurrent Neural Net (Simple RNN)
- Linear Regression
- Random Walk

## Datasets
We used a combination of public sentiment data from Google Trends and Bitcoin prices from 2016-2020 as training for our LSTM model.  

The datasets are separated into two files:
- BTC-USD_train.csv  
  whats contained in the dataset...??
- BTC-USD_test.csv  
  whats contained in the dataset...??

## Setup

### 1. Requirements and Installation

#### Python and Jupyter
Install [Jupyter Notebook](https://test-jupyter.readthedocs.io/en/latest/install.html) through Anaconda, and follow the installation guide. It will include the installation of Python and Jupyter on your machine.

#### Packages for Anaconda
Install these packages in [Anaconda](https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/) in order the code to run:
- Numpy
- Matplotlib.pyplot
- Pandas
- Sklearn.preprocessing
  - ~~MinMaxScaler~~
- Tensorflow.keras
  - ~~Sequential~~
  - ~~Layers~~
- Tensorflow.keras.layers
  - ~~Dense~~
  - ~~LSTM~~
  - ~~Dropout~~
- Sklearn.linear_model
  - ~~LinearRegression~~
  
#### Source Code and Datasets
Download the Jupyter Notebook file `FinalCode-BTC-prediction.ipynb` [here](https://github.com/hkim42/CS542_Team4_BTCprediction/tree/master/Jupyter%20Notebook%20files).  
Download the training dataset `BTC-USD_train.csv` [here](https://github.com/hkim42/CS542_Team4_BTCprediction/tree/master).  
Download the testing dataset `BTC-USD_test.csv` [here](https://github.com/hkim42/CS542_Team4_BTCprediction/tree/master).

### 2. Running the code
1. Run the Jupyter Notebook file by clicking the Jupyter Notebook icon through Anaconda or by running the command `jupyter notebook` in the terminal.
2. Open `FinalCode-BTC-prediction.ipynb` by navigating to the location folder where you downloaded the file through the Jupyter Notebook dashboard.
3. Press the ⏩️ button in the middle of the toolbar to run the notebook.
