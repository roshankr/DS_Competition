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
import heapq
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
#import xgboost as xgb
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
#Airbnb_New_User_Bookings Classification
########################################################################################################################
#--------------------------------------------Algorithm : Random Forest :------------------------------------------------
#Random Forest :
#--------------------------------------------Algorithm : XGB------------------------------------------------------------
#XGB :

#--------------------------------------------Suggestions, Ideas---------------------------------------------------------
#Suggestions, Ideas
#--------------------------------------------with only 7K records-------------------------------------------------------

########################################################################################################################
#NDCG Scores
########################################################################################################################
def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]

    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))

        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k=5, method=1):

    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.

    return dcg_at_k(r, k, method) / dcg_max

def score_predictions(preds, truth, n_modes=5):
    """
    preds: pd.DataFrame
      one row for each observation, one column for each prediction.
      Columns are sorted from left to right descending in order of likelihood.
    truth: pd.Series
      one row for each obeservation.
    """
    test = truth.reset_index()
    test.columns = ['index','1']

    assert(len(preds)==len(truth))
    r = pd.DataFrame(0, index=preds.index, columns=preds.columns, dtype=np.float64)

    for col in preds.columns:
        r[col] = np.where(preds[col] == test['1'],1,0)

    score = pd.Series(r.apply(ndcg_at_k, axis=1, reduce=True), name='score')
    return score

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
#Get the best 5 predictions
########################################################################################################################
def get_best_five(pred,type_val):

    if type_val:

        test = pd.DataFrame(pred).stack().reset_index()
        test.columns = ['row','col','val']
        test['val'] = test['val'].astype(float)

        test = test.sort(columns=['row','val'], ascending=[True,False])
        test = test.groupby(['row']).head(5)
        test = test.drop(['val'], axis = 1)

        test1 = test.groupby(['row']).nth(0)
        test1['1'] = test.groupby(['row']).nth(1)
        test1['2'] = test.groupby(['row']).nth(2)
        test1['3'] = test.groupby(['row']).nth(3)
        test1['4'] = test.groupby(['row']).nth(4)
        test = test1
    else:
        pred = pd.DataFrame(pred)
        pred['actual'] = Actual_DS1
        pred = pred.set_index('actual')
        test = pd.DataFrame(pred).stack().reset_index()
        test.columns = ['id','country','val']
        test['country'] = lbl_y.inverse_transform(test['country'])
        test = test.sort_index()

        test['val'] = test['val'].astype(float)
        test = test.sort(columns=['id','val'], ascending=[True,False])

        test = test.groupby(['id']).head(5)
        test = test.drop(['val'], axis = 1)

        test = test.reset_index(drop=True).set_index('id')

    return test

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
        # temp = pd.DataFrame(y_pred)
        y_pred = get_best_five(y_pred,type_val=True)

        scores.append(score_predictions(y_pred, pd.DataFrame(y_test)).mean())

        # temp = pd.concat([temp,y_pred],axis=1)
        # temp['test'] = y_test
        # temp['val'] = score_predictions(y_pred, pd.DataFrame(y_test))
        # temp.to_csv(file_path+'temp.csv')

        print(" %d-iteration... %s " % (i,scores))

        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Age_DS  cleansing , feature scalinng , splitting
########################################################################################################################
def Cleanup_Age(Age_DS):

    print("***************Starting Data cleansing - Age ***************")

    Age_DS = Age_DS.drop(['year'], axis = 1)
    Age_DS['age_bucket']  = Age_DS['age_bucket'].replace(to_replace='100+', value='100-2015')
    Age_DS['gender']  = Age_DS['gender'].replace(to_replace='male', value='MALE')
    Age_DS['gender']  = Age_DS['gender'].replace(to_replace='female', value='FEMALE')

    New_Age_DS = pd.DataFrame()
    New_Age_DS['age_bucket'] = Age_DS['age_bucket'][Age_DS['country_destination']=='AU']
    New_Age_DS['gender'] = Age_DS['gender'][Age_DS['country_destination']=='AU']

    test = np.unique(list(np.unique(Age_DS['country_destination'].values)))

    for country in np.unique(list(np.unique(Age_DS['country_destination'].values))):

        Temp_DS = Age_DS[Age_DS['country_destination']==country]
        Temp_DS = Temp_DS.drop(['country_destination'], axis = 1)
        Temp_DS.columns = ['age_bucket','gender',country+'_pop']
        New_Age_DS = New_Age_DS.merge(Temp_DS,on=['age_bucket','gender'],how='left')
        New_Age_DS[country+'_pop'] = New_Age_DS[country+'_pop'] / (Temp_DS[country+'_pop'].sum())

    New_Age_DS['age_1'] = New_Age_DS['age_bucket'].str.split('-').str.get(0)
    New_Age_DS['age_2'] = New_Age_DS['age_bucket'].str.split('-').str.get(1)
    New_Age_DS = New_Age_DS.drop(['age_bucket'], axis = 1)

    Test_DS = pd.DataFrame()
    Temp_DS = pd.DataFrame()
    for i in range(len(New_Age_DS)):
        count = int(New_Age_DS.ix[i,'age_2']) - int(New_Age_DS.ix[i,'age_1']) + 1
        Temp_DS = pd.DataFrame([New_Age_DS.ix[i,]]*count).reset_index(drop=True).reset_index()
        Temp_DS['age'] = Temp_DS['age_1'].astype(int)+Temp_DS['index']
        Test_DS = Test_DS.append(Temp_DS)

    Test_DS = Test_DS.drop(['index','age_1','age_2'], axis = 1)

    #pd.DataFrame(Test_DS).to_csv(file_path+'Test_DS.csv')

    return Test_DS
########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS, Age_DS, Session_DS):

    print("***************Starting Data cleansing***************")

    global  Train_DS1,Actual_DS1, lbl_y

    ##----------------------------------------------------------------------------------------------------------------##
    y = Train_DS.country_destination.values
    lbl_y = preprocessing.LabelEncoder()
    lbl_y.fit(list(y))
    y = lbl_y.transform(y)
    #print(pd.DataFrame(y).head(), pd.DataFrame(lbl_y.inverse_transform(y)).head())

    Train_DS = Train_DS.drop(['country_destination'], axis = 1)
    ##----------------------------------------------------------------------------------------------------------------##
    # Clean up Age DS
    Age_DS = Cleanup_Age(Age_DS)
    Train_DS = Train_DS.merge(Age_DS, on=['age','gender'], how='left')
    Actual_DS = Actual_DS.merge(Age_DS, on=['age','gender'], how='left')

    ##----------------------------------------------------------------------------------------------------------------##
    # Merge the session data with Train and Test
    Train_DS = Train_DS.merge(Session_DS, on= 'id', how='left')
    Actual_DS = Actual_DS.merge(Session_DS, on= 'id', how='left')
    ##----------------------------------------------------------------------------------------------------------------##
    #Clean up date_account_created
    print("Starting Date conversion")
    temp = pd.DatetimeIndex(Train_DS['date_account_created'])
    Train_DS['Month'] = temp.month
    Train_DS['Year'] = temp.year
    Train_DS['Day'] = temp.day
    Train_DS['Dayofyear'] = temp.dayofyear
    Train_DS['Weekyear'] = temp.weekofyear

    temp = pd.DatetimeIndex(Actual_DS['date_account_created'])
    Actual_DS['Month'] = temp.month
    Actual_DS['Year'] = temp.year
    Actual_DS['Day'] = temp.day
    Actual_DS['Dayofyear'] = temp.dayofyear
    Actual_DS['Weekyear'] = temp.weekofyear

    Train_DS = Train_DS.drop(['date_account_created'], axis = 1)
    Actual_DS = Actual_DS.drop(['date_account_created'], axis = 1)
    ##----------------------------------------------------------------------------------------------------------------##
    #Clean up timestamp_first_active

    Train_DS['timestamp_first_active'] = Train_DS['timestamp_first_active'].astype(str)
    Train_DS['first_active_YY'] = Train_DS['timestamp_first_active'].str[:4].astype(int)
    Train_DS['first_active_MM'] = Train_DS['timestamp_first_active'].str[4:6].astype(int)
    Train_DS['first_active_DD'] = Train_DS['timestamp_first_active'].str[6:8].astype(int)

    Actual_DS['timestamp_first_active'] = Actual_DS['timestamp_first_active'].astype(str)
    Actual_DS['first_active_YY'] = Actual_DS['timestamp_first_active'].str[:4].astype(int)
    Actual_DS['first_active_MM'] = Actual_DS['timestamp_first_active'].str[4:6].astype(int)
    Actual_DS['first_active_DD'] = Actual_DS['timestamp_first_active'].str[6:8].astype(int)

    Train_DS = Train_DS.drop(['timestamp_first_active'], axis = 1)
    Actual_DS = Actual_DS.drop(['timestamp_first_active'], axis = 1)
    ##----------------------------------------------------------------------------------------------------------------##
    #Clean up date_first_booking

    # Train_DS['date_first_booking'] = Train_DS['date_first_booking'].fillna(0)
    # temp = pd.DatetimeIndex(Train_DS['date_first_booking'])
    # Train_DS['first_booking_MM'] = temp.month
    # Train_DS['first_booking_YY'] = temp.year
    # Train_DS['first_booking_DD'] = temp.day
    # Train_DS['first_booking_Dayofyear'] = temp.dayofyear
    # Train_DS['first_booking_Weekyear'] = temp.weekofyear
    #
    # Actual_DS['date_first_booking'] = Actual_DS['date_first_booking'].fillna(0)
    # temp = pd.DatetimeIndex(Actual_DS['date_first_booking'])
    # Actual_DS['first_booking_MM'] = temp.month
    # Actual_DS['first_booking_YY'] = temp.year
    # Actual_DS['first_booking_DD'] = temp.day
    # Actual_DS['first_booking_Dayofyear'] = temp.dayofyear
    # Actual_DS['first_booking_Weekyear'] = temp.weekofyear

    # Train_DS['date_first_booking'] = np.where(Train_DS['date_first_booking']==0, 0, 1)
    # Actual_DS['date_first_booking'] = np.where(Actual_DS['date_first_booking']==0, 0, 1)

    Train_DS = Train_DS.drop(['date_first_booking'], axis = 1)
    Actual_DS = Actual_DS.drop(['date_first_booking'], axis = 1)

    ##----------------------------------------------------------------------------------------------------------------##
    #Gender Field
    # gender={'MALE':1, 'FEMALE':2, 'OTHER':3, '-unknown-':4}
    # Train_DS['gender'] = Train_DS['gender'].replace(gender).astype(int)
    # Actual_DS['gender'] = Actual_DS['gender'].replace(gender).astype(int)

    #no benefit...ignored
    # Train_DS['gender']  = Train_DS['gender'].replace(to_replace='-unknown-', value='FEMALE')
    # Actual_DS['gender'] = Actual_DS['gender'].replace(to_replace='-unknown-', value='FEMALE')

    ##----------------------------------------------------------------------------------------------------------------##
    #Age
    Train_DS['age'] = Train_DS['age'].fillna(0)
    Actual_DS['age'] = Actual_DS['age'].fillna(0)

    Train_DS['age'] = np.where(np.logical_or(Train_DS['age']<14, Train_DS['age']>80), 0, Train_DS['age'])
    Actual_DS['age'] = np.where(np.logical_or(Actual_DS['age']<14, Actual_DS['age']>80), 0, Actual_DS['age'])

    ##----------------------------------------------------------------------------------------------------------------##
    #signup_method
    # signup={'basic':1, 'facebook':2, 'google':3, 'weibo' :4}
    # Train_DS['signup_method'] = Train_DS['signup_method'].replace(signup).astype(int)
    # Actual_DS['signup_method'] = Actual_DS['signup_method'].replace(signup).astype(int)

    ##----------------------------------------------------------------------------------------------------------------##
    #signup_app
    # signup={'Android':1, 'iOS':2, 'Moweb':3, 'Web':4}
    # Train_DS['signup_app'] = Train_DS['signup_app'].replace(signup).astype(int)
    # Actual_DS['signup_app'] = Actual_DS['signup_app'].replace(signup).astype(int)

    ##----------------------------------------------------------------------------------------------------------------##
    #signup_flow ** Nothing to do as of now
    ##----------------------------------------------------------------------------------------------------------------##
    #Delete the Id
    Actual_DS1 = Actual_DS['id']
    Train_DS = Train_DS.drop(['id'], axis = 1)
    Actual_DS = Actual_DS.drop(['id'], axis = 1)

    ##----------------------------------------------------------------------------------------------------------------##

    #All remaining must be label encoded
    cols = ['language','affiliate_channel','affiliate_provider','first_affiliate_tracked','first_device_type','first_browser','gender','signup_method','signup_app']
    # for i in range(len(cols)):
    #     lbl = preprocessing.LabelEncoder()
    #     lbl.fit((list(Train_DS[cols[i]].astype(str)) + list(Actual_DS[cols[i]].astype(str))))
    #     Train_DS[cols[i]] = lbl.transform(Train_DS[cols[i]].astype(str))
    #     Actual_DS[cols[i]] = lbl.transform(Actual_DS[cols[i]].astype(str))

    # Train_DS = pd.DataFrame(Train_DS)
    # Actual_DS = pd.DataFrame(Actual_DS)

    #Create Dummy for all other var's
    New_DS = pd.concat([Train_DS, Actual_DS])
    for f in cols:
        New_DS_dummy = pd.get_dummies(New_DS[f], prefix=f)
        New_DS = New_DS.drop([f], axis=1)
        New_DS = pd.concat((New_DS, New_DS_dummy), axis=1)

    Train_DS = New_DS.head(len(Train_DS))
    Actual_DS = New_DS.tail(len(Actual_DS))

    ##----------------------------------------------------------------------------------------------------------------##
    print("Any scaling , log transformations")

    Train_DS = Train_DS.replace([np.inf, -np.inf], np.nan)
    Actual_DS = Actual_DS.replace([np.inf, -np.inf], np.nan)

    Train_DS = Train_DS.fillna(0)
    Actual_DS = Actual_DS.fillna(0)

    # Train_DS = np.array(np.log(1 + Train_DS))
    # Actual_DS = np.array(np.log(1 + Actual_DS))

    # Train_DS, y = shuffle(Train_DS, y)
    #
    # #Setting Standard scaler for data
    # stdScaler = StandardScaler(with_mean=True, with_std=True)
    # stdScaler.fit(Train_DS,y)
    # Train_DS = stdScaler.transform(Train_DS)
    # Actual_DS = stdScaler.transform(Actual_DS)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    # pd.DataFrame(Train_DS).to_csv(file_path+'Train_DS_50000.csv')
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

    ##----------------------------------------------------------------------------------------------------------------##
        #CV: 0.79185 (with 50 K) - n_estimators = 200
        #CV: 0.79285 (with 50 K) - n_estimators = 200 - with Age Bkt features
        #CV: 0.80738 (with 50 K) - n_estimators = 200 - with Age Bkt and Session (Action_Type dummy) features  *********
        #CV: 0.80901 (with 50 K) - n_estimators = 200 - with Age Bkt and Session (Action_Type) & and no log, scaling
    ##----------------------------------------------------------------------------------------------------------------##
        #CV: 0.78790 (with 50 K) - n_estimators = 100
        #CV: 0.78912 (with 50 K) - n_estimators = 100 - with Age Bkt features
        #CV: 0.80382 (with 50 K) - n_estimators = 100 - with Age Bkt and Session (Action_Type dummy) features
        #CV: 0.80382 (with 50 K) - n_estimators = 100 - with Age Bkt and Session (Action_Type) & and no log, scaling
    ##----------------------------------------------------------------------------------------------------------------##
        clf = RandomForestClassifier(n_jobs=-1, n_estimators=200)

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        sys.exit(0)

        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)
    pred_Actual = get_best_five(pred_Actual,type_val=True)
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

        param_grid = {'n_estimators': [50],
                      'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,19,20,40,80],
                      'min_child_weight': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,40,80],
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
        #CV: 0.78526434774405007 (full set)
        #CV: 0.824999 (100k set - with Age set up, all dummy)
        clf = xgb.XGBClassifier(n_estimators=75,nthread=8)

        # clf = xgb.XGBClassifier(max_depth=6, learning_rate=0.25, n_estimators=43,
        #             objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0)

        ##----------------------------------------------------------------------------------------------------------------##
        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        sys.exit(0)

        X_train = np.array(Train_DS)
        Y_train = np.array(y)

        clf.fit(X_train, Y_train)

    X_Actual = np.array(Actual_DS)

    #Predict actual model
    pred_Actual = clf.predict_proba(X_Actual)

    pred_Actual = get_best_five(pred_Actual,type_val=False)
    print("Actual Model predicted")

    #Get the predictions for actual data set
    pred_Actual.to_csv(file_path+'output/Submission_Roshan_xgb_1.csv', index_label='id')

    print("***************Ending XGB Classifier***************")
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, Train_DS1

    # Mlogloss
    #Mlogloss_scorer = metrics.make_scorer(multiclass_log_loss, greater_is_better = False)

    random.seed(42)
    np.random.seed(42)

    if(platform.system() == "Windows"):

        file_path = 'C:/Python/Others/data/Kaggle/Airbnb_New_User_Bookings/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Airbnb_New_User_Bookings/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS      = pd.read_csv(file_path+'train_users_2.csv',sep=',')
    Actual_DS     = pd.read_csv(file_path+'test_users.csv',sep=',')
    Sample_DS     = pd.read_csv(file_path+'sample_submission_NDF.csv',sep=',')
    Age_DS        = pd.read_csv(file_path+'age_gender_bkts.csv',sep=',')
    Session_DS    = pd.read_csv(file_path+'New_Session_DS_action_type.csv',sep=',')
    #Session_DS    = pd.read_csv(file_path+'New_Session_DS_action_detail.csv',sep=',',index_col=0)

    # Train_DS = Train_DS.fillna(0)
    # Actual_DS = Actual_DS.fillna(0)
    # print("TRAIN")
    # print(Train_DS.groupby('age').age.count())
    # print("TEST")
    # print(Actual_DS.groupby('age').age.count())


    Create_file = False
    count = 50000

    ifile = 1

    ##----------------------------------------------------------------------------------------------------------------##
    if Create_file:

        Train_DS = (Train_DS.reindex(np.random.permutation(Train_DS.index))).reset_index(drop=True)
        Train_DS = Train_DS.head(count)
        pd.DataFrame(Train_DS).to_csv(file_path+'train_users_'+str(ifile)+'.csv')

        # Actual_DS = (Actual_DS.reindex(np.random.permutation(Actual_DS.index))).reset_index(drop=True)
        # Actual_DS = Actual_DS.head(count)
        # pd.DataFrame(Actual_DS).to_csv(file_path+'test_users_'+str(ifile)+'.csv')

    else:
        Train_DS    = pd.read_csv(file_path+'train_users_'+str(ifile)+'.csv',sep=',',index_col=0,nrows = count)
        #Actual_DS   = pd.read_csv(file_path+'test_users_'+str(ifile)+'.csv',sep=',',index_col=0,nrows = count)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    ##----------------------------------------------------------------------------------------------------------------##
    Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS, Age_DS, Session_DS)

    pred_Actual  = RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    #pred_Actual  = XGB_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys)