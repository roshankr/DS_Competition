import requests
import numpy as np
import scipy as sp
import sys
import platform
import pandas as pd
from time import time
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier ,ExtraTreesClassifier,AdaBoostClassifier, BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import *
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import re
import random
import warnings
from math import sqrt, exp, log
from csv import DictReader
from sklearn.preprocessing import Imputer
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV, ParameterSampler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint as sp_randint
from sklearn import decomposition, pipeline, metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score,roc_curve,auc
import collections
import ast
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LogisticRegression, \
    Perceptron,RidgeCV, TheilSenRegressor
from datetime import date,timedelta as td,datetime as dt
import datetime
from sklearn.feature_selection import SelectKBest,SelectPercentile, f_classif, GenericUnivariateSelect
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.lda import LDA
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import pickle as pickle
from lasagne.layers import get_output,InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum,adagrad
from lasagne.objectives import *
from nolearn.lasagne import NeuralNet
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from lasagne import layers
from lasagne.nonlinearities import  softmax, rectify
from lasagne.updates import nesterov_momentum,sgd,adagrad,adadelta,rmsprop
from lasagne import nonlinearities as nl
from lasagne.regularization import *
########################################################################################################################
#Walmart Recruiting: Trip Type Classification
########################################################################################################################
#--------------------------------------------Algorithm : Random Forest :------------------------------------------------
#Random Forest :
#--------------------------------------------Algorithm : XGB------------------------------------------------------------
#XGB :

#--------------------------------------------Suggestions, Ideas---------------------------------------------------------
#Suggestions, Ideas
#--------------------------------------------with only 7K records-------------------------------------------------------
# RF : 0.7410 - 7414 (with 7k)
########################################################################################################################

########################################################################################################################
#we find the encode layer from our ae, and use it to define an encoding function - Autoencoder
########################################################################################################################
def get_layer_by_name(net, name):
    for i, layer in enumerate(net.get_all_layers()):
        if layer.name == name:
            return layer, i
    return None, None


def encode_input(encode_layer, X):
    return get_output(encode_layer, inputs=X).eval()

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS):

    print("***************Starting Data cleansing***************")

    global  New_DS1

    y = Train_DS.TripType.values

    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(y))
    y = lbl.transform(y)

    Train_DS = Train_DS.drop(['TripType'], axis = 1)
    ##----------------------------------------------------------------------------------------------------------------##
    #Label Encode Weekday
    lbl = preprocessing.LabelEncoder()
    lbl.fit((list(Train_DS['Weekday'].astype(str)) + list(Actual_DS['Weekday'].astype(str))))
    Train_DS['Weekday'] = lbl.transform(Train_DS['Weekday'].astype(str))
    Actual_DS['Weekday'] = lbl.transform(Actual_DS['Weekday'].astype(str))

    #weekday one hot encoding
    print("weekday one hot encoding")
    New_DS = pd.concat([Train_DS, Actual_DS])
    dummies = pd.get_dummies(New_DS['Weekday'])
    cols_new = [ 'Weekday'+"_"+str(s) for s in list(dummies.columns)]
    New_DS[cols_new] = dummies
    Train_DS = New_DS.head(len(Train_DS))
    Actual_DS = New_DS.tail(len(Actual_DS))
    ##----------------------------------------------------------------------------------------------------------------##
    #Merge HighLow contrib

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    Train_DS = Train_DS.merge(HLContrib_DS,on='VisitNumber',how='left')
    Actual_DS = Actual_DS.merge(HLContrib_DS,on='VisitNumber',how='left')

    ##----------------------------------------------------------------------------------------------------------------##
    #Deleting any features during testing
    #ifyou want to delete main Fn
    # test = Train_DS.head()
    # test = test.ix[:,'FN_0':'FN_9999'].columns
    # Train_DS = Train_DS.drop(test, axis = 1)
    # Actual_DS = Actual_DS.drop(test, axis = 1)

    # #ifyou want to delete 1000 Fn
    test = Train_DS.head()
    test = test.ix[:,'FinelineNumber_1000_1.0':'FinelineNumber_1000_9998.0'].columns
    Train_DS = Train_DS.drop(test, axis = 1)
    Actual_DS = Actual_DS.drop(test, axis = 1)
    #
    # #ifyou want to delete 1000 Upc
    test = Train_DS.head()
    test = test.ix[:,'Upc_1000_3082.0':'Upc_1000_775014200016.0'].columns
    Train_DS = Train_DS.drop(test, axis = 1)
    Actual_DS = Actual_DS.drop(test, axis = 1)

    # #Delete only if DD with similarity matrix included
    test = Train_DS.head()
    test = test.ix[:,'DD_buy_0':'DD_ret_WIRELESS'].columns
    Train_DS = Train_DS.drop(test, axis = 1)
    Actual_DS = Actual_DS.drop(test, axis = 1)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    ##----------------------------------------------------------------------------------------------------------------##
    print("Any scaling , log transformations")

    Train_DS = Train_DS.replace([np.inf, -np.inf], np.nan)
    Actual_DS = Actual_DS.replace([np.inf, -np.inf], np.nan)

    Train_DS = Train_DS.fillna(0)
    Actual_DS = Actual_DS.fillna(0)

    New_DS1 = list()
    New_DS1 = list(Train_DS['VisitNumber'])
    New_DS1.extend(list(Actual_DS['VisitNumber']))

    Train_DS = Train_DS.drop(['VisitNumber'], axis = 1)
    Actual_DS = Actual_DS.drop(['VisitNumber'], axis = 1)

    Train_DS = np.array(np.log(100+ Train_DS))
    Actual_DS = np.array(np.log(100+ Actual_DS))

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    print("***************Ending Data cleansing***************")

    return Train_DS , Actual_DS

#########################################################################################################################
#Neural Network Classifier 1
########################################################################################################################
def NN_AutoEncoder(Train_DS , Actual_DS, Grid):

    print("***************Starting NN AutoEncoder***************")
    t0 = time()

    if Grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

    else:
       #y = y.reshape((-1, 1))

       New_DS = np.append(np.array(Train_DS), np.array(Actual_DS),0)
       print(np.shape(New_DS))

       learning_rate = theano.shared(np.float32(0.1))

       New_DS = New_DS.astype('float32')
       num_units = 1000

       #Define Model parms - 2 hidden layers
       clf = NeuralNet(
        	layers=[
                    ('input', InputLayer),
                    ('hidden1', DenseLayer),
                    ('hidden2', DenseLayer),
                    ('encoder', DenseLayer),
                    ('hidden3', DenseLayer),
                    ('hidden4', DenseLayer),
                    ('output', DenseLayer),
       		       ],

       ##-------------------------------------------------------------------------------------------------------------##
       #Input (Input Layer) , Hidden and Output (Dense Layers) parameters

       # Layers:-  http://lasagne.readthedocs.org/en/latest/modules/layers.html
       ##### Network input #####
       # Input Layer - This layer holds a symbolic variable that represents a network input.

       ##### Dense Layer #####
       # DenseLayer  - A fully connected layer.
       # NINLayer    - Network-in-network layer.

       ##### Noise layer #####
       # DropoutLayer       - Dropout layer.
       # dropout            - alias of DropoutLayer
       # GaussianNoiseLayer - Gaussian noise layer.
       ##-------------------------------------------------------------------------------------------------------------##
       # nonlinearity - Non-linear activation functions for artificial neurons.
       # http://lasagne.readthedocs.org/en/latest/modules/nonlinearities.html
       # sigmoid(x)               - Sigmoid activation function (for binary classification)
       # softmax(x)               - Softmax activation function (for multi class classification)
       # tanh(x)                  - Tanh activation function
       # ScaledTanH               - Scaled Tanh activation function
       # rectify(x)               - Rectify activation function (ReLU - ln(1 + e exp(x) )
       # LeakyRectify([leakiness] - Leaky rectifier
       # leaky_rectify(x)         - Instance of LeakyRectify with leakines
       # very_leaky_rectify(x)    - Instance of LeakyRectify with leakiness
       # elu(x)                   - Exponential Linear Unit ( e exp(x) - 1)
       # softplus(x)              - Softplus activation function log(1 + e exp(x)
       # linear(x)                - Linear activation function f(x)=x
       # identity(x)              - Linear activation function f(x)=x
       # x = The activation (the summed, weighted input of a neuron)
       # Default non-linearity is "linear"
       ##-------------------------------------------------------------------------------------------------------------##

       input_shape=(None, New_DS.shape[1]),

       hidden1_num_units=num_units,
       hidden1_nonlinearity=nl.sigmoid,

       hidden2_num_units=num_units,
       hidden2_nonlinearity=nl.sigmoid,

       encoder_num_units=100,
       encoder_nonlinearity=nl.linear,

       hidden3_num_units=num_units,
       hidden3_nonlinearity=nl.sigmoid,

       hidden4_num_units=num_units,
       hidden4_nonlinearity=nl.sigmoid,

       output_nonlinearity=nl.linear,
       output_num_units=New_DS.shape[1],

       # optimization method:
       ##-------------------------------------------------------------------------------------------------------------##
       #Create update expressions for training, i.e., how to modify the parameters at each training step
       # http://lasagne.readthedocs.org/en/latest/modules/updates.html
       # sgd               - Stochastic Gradient Descent (SGD) updates
       # momentum          - Stochastic Gradient Descent (SGD) updates with momentum
       # nesterov_momentum - Stochastic Gradient Descent (SGD) updates with Nesterov momentum
       # adagrad	       - Adagrad updates
       # rmsprop	       - RMSProp updates
       # adadelta          - Adadelta updates
       # adam              - Adam updates
       ##-------------------------------------------------------------------------------------------------------------##
       #update=nesterov_momentum,
       update=adagrad,
       update_learning_rate=0.01,
       #update_momentum=0.9,

       ##-------------------------------------------------------------------------------------------------------------##
       # Used for building loss expressions for training or validating a neural network.
       # http://lasagne.readthedocs.org/en/latest/modules/objectives.html
       # binary_crossentropy      - Computes log loss for binary classification
       # categorical_crossentropy - Computes the  log loss for multi-class classification probs and softmax output units
       # squared_error            - Computes the element-wise squared difference between two tensors (regression)
       # binary_hinge_loss        - Computes the binary hinge loss between predictions and targets.
       # multiclass_hinge_loss    -  Computes the multi-class hinge loss between predictions and targets.
       # Deaflt - squared_error if regression else categorical_crossentropy
       ##-------------------------------------------------------------------------------------------------------------##
       #objective_loss_function = squared_error,

       ##-------------------------------------------------------------------------------------------------------------##
       max_epochs=5,
       eval_size=0.002,
       regression=True,
       verbose=1,
       )

    #Train_DS, y = shuffle(Train_DS, y, random_state=123)
    clf.fit(New_DS, New_DS)

    #dup the auto encoder model to a file
    #pickle.dump(clf, open(file_path+'Autoencoder_dump.pkl','w'))

    #Predict actual model
    pred_Actual = clf.predict(New_DS)
    print("Actual Model predicted")

    ## we find the encode layer from our ae, and use it to define an encoding function
    print("get encode layer")
    encode_layer, encode_layer_index = get_layer_by_name(clf, 'encoder')

    print("get X_encoded")
    X_encoded = encode_input(encode_layer, New_DS)

    print("merge with visitnumber")
    Temp_DS = pd.concat([pd.DataFrame(New_DS1), pd.DataFrame(X_encoded)],axis=1)

    print(np.shape(Temp_DS))

    pd.DataFrame(Temp_DS).to_csv(file_path+'Autoencoder_output.csv')

    print("***************Ending NN AutoEncoder***************")

    return X_encoded

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, Train_DS1, Featimp_DS, Mlogloss_scorer, HLContrib_DS

    random.seed(21)
    np.random.seed(21)

    if(platform.system() == "Windows"):

        file_path = 'C:/Python/Others/data/Kaggle/Walmart_Recruiting_TTC/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Walmart_Recruiting_TTC/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    # Train_DS      = pd.read_csv(file_path+'train_Grouped5_withFNnumber.csv',sep=',')
    # Actual_DS     = pd.read_csv(file_path+'test_Grouped5_withFNnumber.csv',sep=',')
    Sample_DS     = pd.read_csv(file_path+'sample_submission.csv',sep=',')
    HLContrib_DS  = pd.read_csv(file_path+'High_Lowest_Contributors_utilities.csv',sep=',',index_col=0)

    Create_file = True
    count = 20000

    ifile = 5

    if Create_file:
        Train_DS   = pd.read_csv(file_path+'train_Grouped_withFNnumber_'+str(ifile)+'.csv',sep=',')
        Actual_DS  = pd.read_csv(file_path+'test_Grouped_withFNnumber_'+str(ifile)+'.csv',sep=',')
        Train_DS = Train_DS.drop(['Unnamed: 0'], axis = 1)
        Actual_DS = Actual_DS.drop(['Unnamed: 0'], axis = 1)

        print(np.shape(Train_DS))
        print(np.shape(Actual_DS))
    ##----------------------------------------------------------------------------------------------------------------##
        # Train_DS = (Train_DS.reindex(np.random.permutation(Train_DS.index))).reset_index(drop=True)
        # Train_DS = Train_DS.head(count)
        # pd.DataFrame(Train_DS).to_csv(file_path+'train_Grouped_withFNnumber_temp_'+str(ifile)+'.csv')
        #
        # Actual_DS = (Actual_DS.reindex(np.random.permutation(Actual_DS.index))).reset_index(drop=True)
        # Actual_DS = Actual_DS.head(count)
        # pd.DataFrame(Actual_DS).to_csv(file_path+'test_Grouped_withFNnumber_temp_'+str(ifile)+'.csv')
        #
        # print(np.shape(Train_DS))
        # print(np.shape(Actual_DS))

    else:
        Train_DS   = pd.read_csv(file_path+'train_Grouped_withFNnumber_temp_'+str(ifile)+'.csv',sep=',', index_col=0,nrows = count).reset_index(drop=True)
        Actual_DS  = pd.read_csv(file_path+'test_Grouped_withFNnumber_temp_'+str(ifile)+'.csv',sep=',', index_col=0,nrows = count).reset_index(drop=True)
        Train_DS = Train_DS.drop(['Unnamed: 0.1'], axis = 1)
        Actual_DS = Actual_DS.drop(['Unnamed: 0.1'], axis = 1)

    ##----------------------------------------------------------------------------------------------------------------##
    Train_DS , Actual_DS =  Data_Munging(Train_DS,Actual_DS)

    ##----------------------------------------------------------------------------------------------------------------##
    pred_Actual = NN_AutoEncoder(Train_DS , Actual_DS, Grid=False)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)