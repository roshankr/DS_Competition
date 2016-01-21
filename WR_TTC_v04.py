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
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.lda import LDA
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
sys.path.append('C:\\Python34\\Lib\\site-packages\\xgboost')
import xgboost as xgb
# from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, ReshapeLayer,LSTMLayer,RecurrentLayer
# from lasagne.updates import nesterov_momentum,adagrad
# from lasagne.objectives import binary_crossentropy, categorical_crossentropy
# from nolearn.lasagne import NeuralNet
# import theano
# from theano import tensor as T
# from theano.tensor.nnet import sigmoid
# from lasagne import layers
# from lasagne.nonlinearities import  softmax, rectify
# from lasagne.updates import nesterov_momentum,sgd,adagrad,adadelta,rmsprop
# from lasagne import nonlinearities as nl
# from nolearn.lasagne import BatchIterator
# from lasagne.regularization import *
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
#Class AdjustVariable for NN
########################################################################################################################
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

########################################################################################################################
#Class EarlyStopping for NN
########################################################################################################################
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

#########################################################################################################################
def float32(k):
    return np.cast['float32'](k)
#########################################################################################################################
#Build Basic Neural Network Model
########################################################################################################################
def build_mlp(input_num_inputs, output_num_units):

    print("***************Starting NN1 Classifier***************")
    #Define Model parms - 2 hidden layers
    clf = NeuralNet(
        	layers=[
                    ('input', InputLayer),
                    ('dropout0', DropoutLayer),
                    ('hidden1', DenseLayer),
                    ('dropout1', DropoutLayer),
                    ('hidden2', DenseLayer),
                    ('dropout2', DropoutLayer),
                    ('hidden3', DenseLayer),
                    ('dropout3', DropoutLayer),
                    ('output', DenseLayer)
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
       # rectify(x)               - Rectify activation function max(0,z) -- (ReLU - ln(1 + e exp(x) )
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
       input_shape=(None, input_num_inputs),
       dropout0_p=0.15,

       hidden1_num_units=500,
       hidden1_nonlinearity=nl.sigmoid,
       dropout1_p=0.20,

       hidden2_num_units=500,
       hidden2_nonlinearity=nl.sigmoid,
       dropout2_p=0.20,

       hidden3_num_units=500,
       hidden3_nonlinearity=nl.sigmoid,
       dropout3_p=0.20,

       output_nonlinearity=softmax,
       output_num_units=output_num_units,

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
       update=adagrad,
       #update=nesterov_momentum,
       update_learning_rate=theano.shared(float32(0.01)),
       #update_momentum=theano.shared(float32(0.9)),

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
       objective_loss_function = categorical_crossentropy,

       ##-------------------------------------------------------------------------------------------------------------##
       max_epochs=500,
       eval_size=0.2,
       #train_split=TrainSplit(eval_size=0.2),
       regression=False,
       verbose=1,

       ##-------------------------------------------------------------------------------------------------------------##
       ## If label encoding is needed while clf.fit() ...label is already encoded in our case
       use_label_encoder=False,

       ## batch_iterator_train default is 128
       batch_iterator_train=BatchIterator(batch_size=128),
       batch_iterator_test=BatchIterator(batch_size=128),

       on_epoch_finished=[
       AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
       #AdjustVariable('update_momentum', start=0.9, stop=0.999),
       EarlyStopping(patience=25)
       ]
       ##-------------------------------------------------------------------------------------------------------------##
       )

    return clf

#########################################################################################################################
#Build Basic Neural Network Model
########################################################################################################################
def build_rnn(input_num_inputs, output_num_units):

    print("***************Starting NN1 Classifier***************")
    #Define Model parms - 2 hidden layers
    clf = NeuralNet(
        	layers=[
                    ('input', InputLayer),
                    ('lstm1', LSTMLayer),
                    ('rshp1', ReshapeLayer),
                    ('hidden1', DenseLayer),
                    ('output', DenseLayer)
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
       # rectify(x)               - Rectify activation function max(0,z) -- (ReLU - ln(1 + e exp(x) )
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
       #Shape input shape(343) * 512 * 37 * 37

       input_shape=(None, input_num_inputs),

       #batchsize, seqlen, _ = input_input_var.shape,
       lstm1_num_units = 512,
       lstm1_nonlinearity=nl.sigmoid,

       rshp1_shape = (-1, 512),

       hidden1_num_units=output_num_units,
       hidden1_nonlinearity=nl.sigmoid,

       output_nonlinearity=softmax,
       output_num_units=output_num_units,

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
       update=adagrad,
       #update=sgd,
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
       objective_loss_function = categorical_crossentropy,

       ##-------------------------------------------------------------------------------------------------------------##
       max_epochs=50,
       eval_size=0.2,
       #train_split=TrainSplit(eval_size=0.2),
       regression=False,
       verbose=1,

       ##-------------------------------------------------------------------------------------------------------------##
       ## If label encoding is needed while clf.fit() ...label is already encoded in our case
       use_label_encoder=False,

       ## batch_iterator_train default is 128
       batch_iterator_train=BatchIterator(batch_size=128),
       batch_iterator_test=BatchIterator(batch_size=128),

       # on_epoch_finished=[
       # AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
       # AdjustVariable('update_momentum', start=0.9, stop=0.999),
       # EarlyStopping(patience=10)
       # ]
       ##-------------------------------------------------------------------------------------------------------------##
       )

    return clf

########################################################################################################################
#Utility function to report best scores
########################################################################################################################
def report(grid_scores, n_top):

    cols_key = []
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]

    for i, score in enumerate(top_scores):
        if( i < 5):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")

        dict1 = collections.OrderedDict(sorted(score.parameters.items()))

        if i==0:
            for key in dict1.keys():
                cols_key.append(key)

            cols_key.append('CV')
            Parms_DF =  pd.DataFrame(columns=cols_key)

        cols_val = []
        for key in dict1.keys():
            cols_val.append(dict1[key])

        cols_val.append(score.mean_validation_score)

        Parms_DF.loc[i] =  cols_val

    return Parms_DF

########################################################################################################################
#multiclass_log_loss
########################################################################################################################

def multiclass_log_loss(y_true, y_pred):
    return log_loss(y_true,y_pred, eps=1e-15, normalize=True )

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid(X, y, clf):

    print("***************Starting Kfold Cross validation***************")

    X =np.array(X)
    scores=[]

    # lbl = preprocessing.LabelEncoder()
    # lbl.fit(list(y))
    # y = lbl.transform(y)

    ss = StratifiedShuffleSplit(y, n_iter=5,test_size=0.2)
    #ss = KFold(len(y), n_folds=5,shuffle=False,indices=None)

    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        #clf.fit(X_train, y_train, early_stopping_rounds=25, eval_metric="mlogloss",eval_set=[(X_test, y_test)])
        clf.fit(X_train, y_train)

        y_pred=clf.predict_proba(X_test)
        scores.append(log_loss(y_test,y_pred, eps=1e-15, normalize=True ))
        print(" %d-iteration... %s " % (i,scores))

        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS):

    print("***************Starting Data cleansing***************")

    global  Train_DS1

    y = Train_DS.TripType.values

    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(y))
    y = lbl.transform(y)

    Train_DS = Train_DS.drop(['TripType'], axis = 1)
    ##----------------------------------------------------------------------------------------------------------------##

    Train_DS['Weektype'] = np.where(np.logical_or(Train_DS['Weekday']=='Saturday' ,Train_DS['Weekday']=='Sunday' ), 1,2)

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

    # Train_DS = Train_DS.merge(Autoencoder_DS,on='VisitNumber',how='left')
    # Actual_DS = Actual_DS.merge(Autoencoder_DS,on='VisitNumber',how='left')

    # newcols = list(HLContrib_DS_2.ix[:,'DD_buy1_0':'DD_ret1_68'].columns)
    # newcols.extend(['VisitNumber'])
    # Train_DS = Train_DS.merge(HLContrib_DS_2[newcols],on='VisitNumber',how='left')
    # Actual_DS = Actual_DS.merge(HLContrib_DS_2[newcols],on='VisitNumber',how='left')

    ##----------------------------------------------------------------------------------------------------------------##
    #Deleting any features during testing
    #ifyou want to delete main Fn
    test = Train_DS.head()
    test = test.ix[:,'FN_0':'FN_9999'].columns
    Train_DS = Train_DS.drop(test, axis = 1)
    Actual_DS = Actual_DS.drop(test, axis = 1)

    #ifyou want to delete 1000 Fn
    test = Train_DS.head()
    test = test.ix[:,'FinelineNumber_1000_1.0':'FinelineNumber_1000_9998.0'].columns
    Train_DS = Train_DS.drop(test, axis = 1)
    Actual_DS = Actual_DS.drop(test, axis = 1)

    #ifyou want to delete 1000 Upc
    test = Train_DS.head()
    test = test.ix[:,'Upc_1000_3082.0':'Upc_1000_775014200016.0'].columns
    Train_DS = Train_DS.drop(test, axis = 1)
    Actual_DS = Actual_DS.drop(test, axis = 1)

    #Delete only if DD with similarity matrix included
    test = Train_DS.head()
    test = test.ix[:,'DD_buy_0':'DD_ret_WIRELESS'].columns
    Train_DS = Train_DS.drop(test, axis = 1)
    Actual_DS = Actual_DS.drop(test, axis = 1)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))
    ##----------------------------------------------------------------------------------------------------------------##

    print("Any scaling , log transformations")

    Actual_DS = Actual_DS.sort(columns='VisitNumber',ascending=True)
    Train_DS, y = shuffle(Train_DS, y)

    Train_DS = Train_DS.drop(['VisitNumber'], axis = 1)
    Actual_DS = Actual_DS.drop(['VisitNumber'], axis = 1)

    Train_DS = Train_DS.replace([np.inf, -np.inf], np.nan)
    Actual_DS = Actual_DS.replace([np.inf, -np.inf], np.nan)

    Train_DS = Train_DS.fillna(0)
    Actual_DS = Actual_DS.fillna(0)

    Train_DS = np.array(np.log(100+ Train_DS))
    Actual_DS = np.array(np.log(100+ Actual_DS))

    #Setting Standard scaler for data
    stdScaler = StandardScaler(with_mean=True, with_std=True)
    stdScaler.fit(Train_DS,y)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

    # Train_DS = np.array(Train_DS)
    # Actual_DS = np.array(Actual_DS)

    #Use PCA for feature extraction
    # pca = PCA(n_components=500)
    # pca.fit(Train_DS,y )
    # Train_DS = pca.transform(Train_DS)
    # Actual_DS = pca.transform(Actual_DS)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    # pd.DataFrame(Train_DS).to_csv(file_path+'Train_DS_50000.csv')
    # sys.exit(0)
    print("***************Ending Data cleansing***************")

    return Train_DS, Actual_DS, y

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid):
    print("***************Starting Random Forest Classifier***************")
    t0 = time()

    if Grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "criterion":['gini', 'entropy'],
                      "max_depth": [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15, None],
                      "max_features": [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15, None,'auto','log2'],
                      "min_samples_split": sp_randint(1, 50),
                      "min_samples_leaf": sp_randint(1, 50),
                      "bootstrap": [True],
                      "oob_score": [True, False]
                     }

        clf = RandomForestClassifier(n_estimators=100,n_jobs=-1)

        # run randomized search
        n_iter_search = 3000
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = 'roc_auc',cv=5)

        start = time()
        clf.fit(Train_DS, y)

        print("RandomizedSearchCV took %.2f seconds for %d candidates"
                " parameter settings." % ((time() - start), n_iter_search))
        report(clf.grid_scores_)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
    else:

        #CV: 1.505327 , 20 K , n_estimators =100 , features = 343 (without FN and Upc and using eucledean for DD)
        clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_split=1,max_features='auto',bootstrap=True,
                                        max_depth = 8, min_samples_leaf = 4,oob_score=True,criterion='entropy')

        #CV: 1.995509 , 20 K , n_estimators =100 , features = 343 (without FN and Upc and using eucledean for DD)
        clf = RandomForestClassifier(n_jobs=-1, n_estimators=100)

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        sys.exit(0)
        #clf = RandomForestClassifier(n_jobs=-1, n_estimators=2000)
        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        clf.fit(Train_DS, y)

        # #
        # feature = pd.DataFrame()
        # feature['imp'] = clf.feature_importances_
        # feature['col'] = Train_DS1.columns
        # feature = feature.sort(['imp'], ascending=False).reset_index(drop=True)
        # print(feature)
        # pd.DataFrame(feature).to_csv(file_path+'feature_imp.csv')

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.VisitNumber.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_rfc_1.csv', index_label='VisitNumber')

    print("***************Ending Random Forest Classifier***************")
    return pred_Actual

########################################################################################################################
#XGB_Classifier
########################################################################################################################
def XGB_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid):

    print("***************Starting XGB Classifier***************")
    t0 = time()

    if Grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        param_grid = {'n_estimators': [25],
                      'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,19,20,40,80,100,200],
                      'min_child_weight': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,40,80,100],
                      'subsample': [0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9,1],
                      'colsample_bytree': [0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9,1],
                      'silent':[True],
                      'gamma':[2,1,0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9]
                     }

        #run randomized search
        n_iter_search = 800
        clf = xgb.XGBClassifier(nthread=8)
        clf = RandomizedSearchCV(clf, param_distributions=param_grid,
                                           n_iter=n_iter_search, scoring = 'log_loss',cv=3)
        start = time()
        clf.fit(np.array(Train_DS), np.array(y))

        print("GridSearchCV completed")
        Parms_DS_Out = report(clf.grid_scores_,n_top=n_iter_search)
        Parms_DS_Out.to_csv(file_path+'Parms_DS_XGB_4.csv')

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        sys.exit(0)
    else:
        ##----------------------------------------------------------------------------------------------------------------##
        #best from grid Search, best n_est=175
        #CV:0.936880  , 20 K , n_estimators =100 , features = 343 (without FN and Upc and using eucledean for DD)*** current best
        clf = xgb.XGBClassifier(n_estimators=100,max_depth=100,learning_rate=0.1,nthread=8,min_child_weight=1,
                             subsample=0.6,colsample_bytree=0.9,silent=True, gamma = 2 )

        ##----------------------------------------------------------------------------------------------------------------##
        #CV: 0.955185 , 20 K , n_estimators =100 , features = 343 (without FN and Upc)
        #CV: 0.935217 , 20 K , n_estimators =100 , features = 343 (without FN and Upc and using eucledean for DD)
        #CV: 0.927019 , 20 K , n_estimators =100 , features = 343 (without FN and Upc and using cos_sim for DD) *****not used ovefitting
        #CV: 0.922370 , 20 K , n_estimators =100 , features = 343 (without FN and Upc and using eucl + cos_sim for DD) *****not used ovefitting

        ##................................................................................................................##
        #CV: 0.942477 , 20 K , n_estimators =100 , features = 343 (without FN and Upc and using eucledean for DD)
        #clf = xgb.XGBClassifier(n_estimators=100,nthread=8)

        ##----------------------------------------------------------------------------------------------------------------##

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        sys.exit(0)

        X_train = np.array(Train_DS)
        Y_train = np.array(y)

        clf.fit(X_train, Y_train)

    X_Actual = np.array(Actual_DS)

    #Predict actual model
    pred_Actual = clf.predict_proba(X_Actual)
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.VisitNumber.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_xgb_6_withFNnumber.csv', index_label='VisitNumber')

    print("***************Ending XGB Classifier***************")
    return pred_Actual

########################################################################################################################
#XGB_Classifier
########################################################################################################################
def XGB_Orig_binlog_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid):

    print("***************Starting XGB binlog Classifier***************")
    t0 = time()

    if Grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")
    else:

        #convert all data frames to numpy arrays for xgb use
        dtrain = xgb.DMatrix(Train_DS, label=y)
        dtest  = xgb.DMatrix(Actual_DS)
        print(len(np.unique(y)))

        #only for cross validation
        # X_train, X_cv, Y_train, Y_cv = train_test_split(Train_DS, y, test_size=0.5, random_state=42)
        # dtrain = xgb.DMatrix(X_train, label=Y_train)
        # dtest  = xgb.DMatrix(X_cv, label=Y_cv)

        # specify parameters
        # param = {'max_depth':14, 'eta':0.01, 'min_child_weight':8,'subsample': 0.9,'colsample_bytree':0.3,
        #      'silent':True, 'gamma': 0.9,'nthread': -1,'objective':'binary:logistic', 'eval_metric':'auc' }

        param = {'nthread': 8,'objective':'multi:softprob','num_class':len(np.unique(y)), 'eval_metric':'mlogloss','silent':True}

        plst = param.items()
        #best with 115 rounds 0.7522
        num_round = 115

        #print ('running cross validation')
        #xgb.cv(param, dtrain, num_round, nfold=2,metrics={'mlogloss'}, seed = 0, show_stdv = False)

        # specify validations set to watch performance
        watchlist  = [(dtest,'eval'), (dtrain,'train')]

        print("Starting training")

        #clf = xgb.train( plst, dtrain, num_round,watchlist,early_stopping_rounds=50)

        clf = xgb.train( plst, dtrain, num_round)
        print("training completed")

        #print "testing"
        pred_Actual = clf.predict(dtest)

    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.VisitNumber.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_xgb_orig_6_withFNnumber.csv', index_label='VisitNumber')

    print("***************Ending XGB Classifier***************")
    return pred_Actual
########################################################################################################################
#Misc Classifier
########################################################################################################################
def Misc_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid):
    print("***************Starting Misc Classifier***************")
    t0 = time()

    if Grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

    else:

        #CV - 0.666186155556
        #CV - 0.6670 - remove date MM/DD/YY and todays difff
        # clf = LogisticRegression()
        # Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        #sys.exit(0)

        #print("Adaboost")
        #CV: 0.7099
        #clf = AdaBoostClassifier(n_estimators=100)

        # print("BaggingClassifier")
        # #CV:
        # clf = BaggingClassifier(n_estimators=100)
        # Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        #
        # print("ExtraTreesClassifier")
        # #CV:2.22
        # clf = ExtraTreesClassifier(n_estimators=100)
        # Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        # print("MultinomialNB")
        # #CV:
        # clf = MultinomialNB()
        # Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        #
        # print("BernoulliNB")
        # #CV:
        # clf = BernoulliNB()
        # Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        #clf = SVC(kernel='rbf', class_weight='auto',C=1e5, gamma= 0.001,probability=True)
        clf = SVC(probability=True)
        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        sys.exit(0)

        clf.fit(Train_DS, y)

        # feature = pd.DataFrame()
        # feature['imp'] = clf.feature_importances_
        # feature['col'] = Train_DS1.columns
        # feature = feature.sort(['imp'], ascending=False).reset_index(drop=True)
        # print(feature)

    #Predict actual model
    pred_Actual = clf.predict(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_Misc_filter_2.csv', index_label='ID')

    print("***************Ending Random Forest Classifier***************")
    return pred_Actual

#########################################################################################################################
#Neural Network Classifier 1
########################################################################################################################
def NN1_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid):

    print("***************Starting NN1 Classifier***************")
    t0 = time()

    if Grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

    else:
        #y = y.reshape((-1, 1))

        learning_rate = theano.shared(np.float32(0.1))

        y = y.astype(np.int32)
        Train_DS = Train_DS.astype('float32')
        Actual_DS = Actual_DS.astype('float32')

        ##----------------------------------------------------------------------------------------------------------------##
        #Best CV's
        #CV:1.011700  ,sigmoid,  max_epochs =15 , Dense = 700,700 (without FN and Upc and using eucledean for DD)
        #CV:1.010100  ,sigmoid,  max_epochs =15 , Dense = 1000,1000 (without FN and Upc and using eucledean for DD)
        #CV:0.963210  ,sigmoid,  max_epochs =265 , Dense = 500,500 (without FN and Upc and using eucledean for DD)
        #CV:0.962284  ,sigmoid,  max_epochs =266 , Dense = 500,500 (without FN and Upc and using eucledean for DD & no np.log )*** current best
        #CV:0.965358  ,sigmoid,  max_epochs =227 , Dense = 500,500 (without FN and Upc and using eucledean for DD & autoencoder)

        ##----------------------------------------------------------------------------------------------------------------##
        clf = build_mlp(Train_DS.shape[1],len(np.unique(y)))

        #clf = build_rnn(Train_DS.shape[1],len(np.unique(y)))

        #Train_DS, y = shuffle(Train_DS, y, random_state=123)
        clf.fit(Train_DS, y)

        # _, X_valid, _, y_valid = clf.train_test_split(Train_DS, y, clf.eval_size)
        # probas = clf.predict_proba(X_valid)[:,0]
        # print("ROC score", metrics.roc_auc_score(y_valid, probas))

        print("done in %0.3fs" % (time() - t0))

        sys.exit(0)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)
    print("Actual Model predicted")

    #Get the predictions for actual data set

    preds = pd.DataFrame(pred_Actual, index=Sample_DS.VisitNumber.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_NN5.csv', index_label='VisitNumber')

    print("***************Ending NN1 Classifier***************")
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, Train_DS1, Featimp_DS, Mlogloss_scorer, HLContrib_DS, HLContrib_DS_2, Autoencoder_DS

    # Mlogloss
    Mlogloss_scorer = metrics.make_scorer(multiclass_log_loss, greater_is_better = False)

    random.seed(42)
    np.random.seed(42)

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
    #HLContrib_DS  = pd.read_csv(file_path+'High_Lowest_Contributors_utilities_new.csv',sep=',',index_col=0)
    #HLContrib_DS  = pd.read_csv(file_path+'High_Lowest_Contributors_utilities.csv',sep=',',index_col=0)
    #Autoencoder_DS  = pd.read_csv(file_path+'Autoencoder_output.csv',sep=',',index_col=0)

    Create_file = False
    count = 5000

    ifile = 5

    if Create_file:
        Train_DS   = pd.read_csv(file_path+'train_Grouped_withFNnumber_'+str(ifile)+'.csv',sep=',')
        Actual_DS  = pd.read_csv(file_path+'test_Grouped_withFNnumber_'+str(ifile)+'.csv',sep=',')

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
        Train_DS    = pd.read_csv(file_path+'train_50000.csv',sep=',',index_col=0,nrows = count)
        Actual_DS   = pd.read_csv(file_path+'test_50000.csv',sep=',',index_col=0,nrows = count)
        #Train_DS = (Train_DS.reindex(np.random.permutation(Train_DS.index))).reset_index(drop=True)

        #Train_DS   = pd.read_csv(file_path+'train_Grouped_withFNnumber_temp_'+str(ifile)+'.csv',sep=',', index_col=0,nrows = count).reset_index(drop=True)
        #Actual_DS  = pd.read_csv(file_path+'test_Grouped_withFNnumber_temp_'+str(ifile)+'.csv',sep=',', index_col=0,nrows = count).reset_index(drop=True)

    ##----------------------------------------------------------------------------------------------------------------##

    Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS)

    pred_Actual = XGB_Orig_binlog_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    #pred_Actual = XGB_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    #pred_Actual = XGB_Classifier1(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    #pred_Actual  = RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    pred_Actual  = Misc_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    #pred_Actual = NN1_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys)