# -*- coding: utf-8 -*-
import requests
import numpy as np
import scipy as sp
import scipy.special as sps
import sys
import platform
import pandas as pd
from time import time
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
import re
import time as tm
import warnings
from math import sqrt, exp, log
from csv import DictReader
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LogisticRegression, BayesianRidge
from scipy.stats import randint as sp_randint
from sklearn import decomposition, pipeline, metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
# import xgboost as xgb
# from lasagne import layers
# from lasagne.nonlinearities import  softmax, rectify
# from lasagne.updates import nesterov_momentum,sgd,adagrad
# from lasagne.nonlinearities import identity
# from nolearn.lasagne import NeuralNet
import collections
from sklearn.cross_validation import train_test_split
import random

########################################################################################################################
#Liberty Mutual Group: Property Inspection Prediction
########################################################################################################################
# This is a try to create a N fold stacking algorithm
# Modeled using https://github.com/emanuele/kaggle_pbr/blob/master/blend.py
# Examples are available in http://mlwave.com/kaggle-ensembling-guide/
# Popular non-linear algorithms for stacking are GBM, KNN, NN, RF and ET.
########################################################################################################################

########################################################################################################################
#Gini Scorer - Scorer
########################################################################################################################
def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

########################################################################################################################
#Normalized Gini Scorer
########################################################################################################################
def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

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
            Parms_DF =  pd.DataFrame(columns=cols_key)

        cols_val = []
        for key in dict1.keys():
            cols_val.append(dict1[key])

        Parms_DF.loc[i] =  cols_val

    return Parms_DF

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid(X, y, clf):

    print("***************Starting Kfold Cross validation*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    X =np.array(X)

    scores=[]
    #ss = StratifiedShuffleSplit(y, n_iter=10,test_size=0.3, random_state=42, indices=None)
    ss = KFold(len(y), n_folds=10,shuffle=True,indices=False)
    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        scores.append(normalized_gini(y_test,y_pred))
        print(" %d-iteration... %s " % (i,scores))
        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation*************** at Time: %s" %(tm.strftime("%H:%M:%S")))

    return np.mean(scores)

########################################################################################################################
#Neural network Regressor
########################################################################################################################
def NN1_Regressor(Train_DS, y, Actual_DS, grid):

    print("***************Starting NN Regressor ***************")

    Train_DS = np.array(Train_DS)
    Actual_DS = np.array(Actual_DS)

    t0 = time()

    if grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

    else:

        y = y.reshape((-1, 1))

        Actual_DS  = Actual_DS.astype('float32')
        Train_DS   = Train_DS.astype('float32')
        y          = y.astype('float32')

        print("Starting NN without grid")

        #cv=0.392 - 0.395 LB: 0.385
       #Define Model parms - 2 hidden layers
        clf = NeuralNet(
        	layers=[
       			    ('input', layers.InputLayer),
       			    ('dropout0', layers.DropoutLayer),
        		    ('hidden1', layers.DenseLayer),
        		    ('dropout1', layers.DropoutLayer),
       		 	    ('hidden2', layers.DenseLayer),
       			    ('dropout2', layers.DropoutLayer),
       			    ('output', layers.DenseLayer),
       		       ],

   	    # layer parameters:
        input_shape=(None, Train_DS.shape[1]),
        dropout0_p=0.15,
        hidden1_num_units=500,
        dropout1_p = 0.25,
        hidden2_num_units=500,
        dropout2_p = 0.25,

        #output_nonlinearity=identity,
        output_nonlinearity=None,
        output_num_units=1,

        #optimization method
        #update=sgd,
        #update=nesterov_momentum,
        update=adagrad,
        update_learning_rate=0.001,
        #update_momentum=0.9,

        eval_size = 0.2,
        regression=True,
        max_epochs=15,
        verbose=1
        )

        clf.fit(Train_DS,y)

        _, X_valid, _, y_valid = clf.train_test_split(Train_DS, y, clf.eval_size)

        y_pred=clf.predict(X_valid)
        score=normalized_gini(y_valid, y_pred)
        print("Best score: %0.3f" % score)

    pred_Actual = clf.predict(Actual_DS)
    print("Actual Model predicted")

    print("***************Ending NN Regressor ***************")
    return pred_Actual

########################################################################################################################
#Random Forest Classifier (around 80%)
#####################################################################################
# ###################################
def Final_Stacking(Train_DS, y, Actual_DS, Sample_DS):

    print("***************Starting Final Stacking*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    t0 = time()

    #Setting Standard scaler for data
    # stdScaler = StandardScaler()
    # stdScaler.fit(Train_DS,y)
    # Train_DS = stdScaler.transform(Train_DS)
    # Actual_DS = stdScaler.transform(Actual_DS)

    #CV: 0.36225ff
    # clf = RandomForestRegressor(n_estimators=100,min_samples_leaf=18,max_features=None,bootstrap=True,
    #                                 min_samples_split=23,max_depth=25)

    # clf=Lasso(alpha=0.02)
    # print("lasso CV")
    # Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

    #clf = xgb.XGBRegressor(n_estimators=2000,max_depth=1,learning_rate=0.01,nthread=4,min_child_weight=7,subsample=1,colsample_bytree=0.7,silent=True,gamma=0.8)

    #cv:0.3872
    #clf = ExtraTreesRegressor(n_estimators=1000,min_samples_leaf=19,max_features=12,bootstrap=True,min_samples_split=1,max_depth=25,n_jobs=-1)

    #################################################################################################################################################
    #CV:
    # clf = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
    # #Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
    # clf.fit(Train_DS, y)
    # Pred_Actual = clf.predict(Actual_DS).reshape(-1,1)
    # preds_RFR = pd.DataFrame(Pred_Actual,columns=Sample_DS.columns[1:]).reset_index().sort(columns='Hazard',ascending= True).reset_index(drop=True).reset_index().sort(columns='index',ascending= True).reset_index(drop=True)
    # preds_RFR = preds_RFR.drop(['Hazard','index'], axis = 1)
    #
    # preds = pd.DataFrame(np.array(preds_RFR), index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    # preds.to_csv(file_path+'output/Submission_Stacking_RFR_1.csv', index_label='Id')
    # print("RFR Actual Model predicted")
    # print(preds_RFR.head(10))
    #
    # sys.exit(0)
    ################################################################################################################################################
    #CV:0.39009654989438813
    # clf = xgb.XGBRegressor(n_estimators=2000,max_depth=2,learning_rate=0.01,nthread=4,min_child_weight=23,subsample=0.9,colsample_bytree=0.2,silent=True,gamma=0.9)
    # #clf = xgb.XGBRegressor(n_estimators=1000)
    # #Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
    # clf.fit(Train_DS, y)
    # Pred_Actual = clf.predict(Actual_DS).reshape(-1,1)
    # preds_XGB = pd.DataFrame(Pred_Actual,columns=Sample_DS.columns[1:]).reset_index().sort(columns='Hazard',ascending= True).reset_index(drop=True).reset_index().sort(columns='index',ascending= True).reset_index(drop=True)
    # preds_XGB = preds_XGB.drop(['Hazard','index'], axis = 1)
    #
    # preds = pd.DataFrame(np.array(preds_XGB), index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    # preds.to_csv(file_path+'output/Submission_Stacking_XGB_1.csv', index_label='Id')
    # print("XGB Actual Model predicted")
    # print(preds_XGB.head(10))
    #################################################################################################################################################
    #CV: 0.3879 , LB:0.382419
    #clf = ElasticNet(alpha=0.1, l1_ratio=0.3)

    #CV:0.3902-.3905 , 0.385
    clf = ElasticNet(alpha=0.1, l1_ratio=0.1)
    clf = BayesianRidge(n_iter=300)
    Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
    clf.fit(Train_DS, y)
    Pred_Actual = clf.predict(Actual_DS).reshape(-1,1)
    preds_ELN = pd.DataFrame(Pred_Actual,columns=Sample_DS.columns[1:]).reset_index().sort(columns='Hazard',ascending= True).reset_index(drop=True).reset_index().sort(columns='index',ascending= True).reset_index(drop=True)
    preds_ELN = preds_ELN.drop(['Hazard','index'], axis = 1)

    preds = pd.DataFrame(np.array(preds_ELN), index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Stacking_ELN_1.csv', index_label='Id')
    print("ELN Actual Model predicted")
    print(preds_ELN.head(10))

    sys.exit(0)
    #######################################################################################################################################

    Pred_Actual = NN1_Regressor(Train_DS, y, Actual_DS, grid= False)
    preds_NNT = pd.DataFrame(Pred_Actual,columns=Sample_DS.columns[1:]).reset_index().sort(columns='Hazard',ascending= True).reset_index(drop=True).reset_index().sort(columns='index',ascending= True).reset_index(drop=True)
    preds_NNT = preds_NNT.drop(['Hazard','index'], axis = 1)

    preds = pd.DataFrame(np.array(preds_NNT), index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Stacking_NNT_1.csv', index_label='Id')
    print("NNT Actual Model predicted")
    print(preds_NNT.head(10))

    pred_Actual = (preds_XGB['level_0'] + preds_ELN['level_0'] + preds_NNT['level_0'])/3

    #pred_Actual = np.power((Pred_Actual *  Pred_Actual1 * Pred_Actual2), (1/3.0))

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Stacking_1.csv', index_label='Id')
    ########################################################################################################################################

    print("***************Ending Final Stacking*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, gini_scorer,Parm_RF_DS, Parm_XGB_DS, num_folds

    num_folds = 5
    # Normalized Gini Scorer
    gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)

    if(platform.system() == "Windows"):
        file_path = 'C:/Python/Others/data/Kaggle/Liberty_Mutual_Group/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Liberty_Mutual_Group/'

    Full_run = True
########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Sample_DS = pd.read_csv(file_path+'sample_submission.csv',sep=',')
    Train_DS =  pd.read_csv(file_path+'Train_DS_Stacking_Nfold_v3_reduced.csv',sep=',', index_col=0)
    Actual_DS =  pd.read_csv(file_path+'Actual_DS_Stacking_Nfold_v3_reduced.csv',sep=',', index_col=0)

    y = Train_DS['y']
    Train_DS = Train_DS.drop(['y'], axis = 1)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    Pred_Actual = Final_Stacking(Train_DS, y, Actual_DS, Sample_DS)
########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)