import requests
import numpy as np
import scipy as sp
import sys
import platform
import pandas as pd
from time import time
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
import re
import warnings
from math import sqrt, exp, log
from csv import DictReader
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from scipy.stats import randint as sp_randint
from sklearn import decomposition, pipeline, metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
#import xgboost as xgb
# from lasagne import layers
# from lasagne.nonlinearities import  softmax, rectify
# from lasagne.updates import nesterov_momentum,sgd,adagrad
# from lasagne.nonlinearities import identity
# from nolearn.lasagne import NeuralNet
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
import collections
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Lasso, ElasticNet,Ridge, SGDRegressor,LogisticRegression,BayesianRidge,\
    ARDRegression,Lars,MultiTaskElasticNet, PassiveAggressiveRegressor
from sklearn.decomposition import PCA

########################################################################################################################
#Liberty Mutual Group: Property Inspection Prediction
########################################################################################################################
#--------------------------------------------Algorithm : Random Forest :------------------------------------------------
#Random Forest :
#1. Run Grid search and got all 15 best parms
#2. Run ensemble for all 15 best parms
#3. Top one got the best CV : [0.37361619065157531] , Pub LB : 0.371182
#4. Tried Log transform on RF , but ended up CV : [0.3724xxx
#5. Tried poly features with interaction only and bias as True and CV: [0.3645xxx
#6. Tried poly features with interaction only and no bias and CV: [0.3647xxx.So removed poly features
#--------------------------------------------Algorithm : XGB------------------------------------------------------------
#XGB :
#1. Run Grid search and got all 15 best parms
#2. Run ensemble for all 15 best parms
#3. 12th parm got the best CV : [0.38855373907119883] , Pub LB : 0.382771
#4. Combined output of all 15 ensembles and tried , Pub LB : 0.380363
#5. submitted the first parm output(supposed to be the best output during ensemble) CV : [0.3855035] , Pub LB : 0.374809
#--------------------------------------------Suggestions, Ideas---------------------------------------------------------
#Suggestions, Ideas
#1. As per the grid output , better to run grid search CV=10 and max possible iterations >100 - TODO
#2. Try doing feature engineering - TODO
#3. Try Graphlab GBM - TODO
#4. Try Vowpal Wobit - TODO
########################################################################################################################
#Gini Scorer
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

            cols_key.append('CV')
            Parms_DF =  pd.DataFrame(columns=cols_key)

        cols_val = []
        for key in dict1.keys():
            cols_val.append(dict1[key])

        cols_val.append(score.mean_validation_score)

        Parms_DF.loc[i] =  cols_val

    return Parms_DF

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid(X, y, clf):

    print("***************Starting Kfold Cross validation***************")
    X =np.array(X)
    scores=[]
    #ss = StratifiedShuffleSplit(y, n_iter=10,test_size=0.3, random_state=42, indices=None)
    ss = KFold(len(y), n_folds=5,shuffle=True,indices=False)
    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        clf.fit(X_train, np.log1p(y_train))
        y_pred=np.expm1(clf.predict(X_test))
        scores.append(normalized_gini(y_test,y_pred))
        print(" %d-iteration... %s " % (i,scores))
        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return np.mean(scores)

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS):

    print("***************Starting Data cleansing***************")

    ####################################################################################################################
    #Hazard Means of Factors
    Train_DS_SUM = Train_DS
    Actual_DS_SUM = Actual_DS

    #Retain the same order even after merge
    Train_DS_SUM = Train_DS_SUM.reset_index()
    Actual_DS_SUM = Actual_DS_SUM.reset_index()

    Factors = list(Train_DS_SUM.select_dtypes(include=['object']).columns)
    Factors.remove('T1_V6')
    Factors.remove('T1_V17')
    Factors.remove('T2_V3')
    Factors.remove('T2_V11')
    Factors.remove('T2_V12')
    Train_DS_SUM_Factors = Train_DS_SUM[list(Factors) + ['Hazard']]

    for feature in Factors:
        Feature_mean = Train_DS_SUM.groupby([feature]).agg({'Hazard': ['sum', 'count']}).reset_index()

        #Feature_median = Train_DS_SUM.groupby([feature])['Hazard'].median().reset_index()
        #Feature_SD = Train_DS_SUM.groupby([feature])['Hazard'].agg(np.std).reset_index()
        #Feature_mode = Train_DS_SUM.groupby([feature])['Hazard'].agg(st.mode).reset_index()

        Feature_mean.columns = [feature,feature+'_sum',feature+'_count']

        #Feature_median.columns = [feature,feature+'_median']
        #Feature_SD.columns = [feature,feature+'_SD']
        #Feature_mode.columns = [feature,feature+'_mode']

        Train_DS_SUM  = Train_DS_SUM.merge(Feature_mean, on=feature)

        # excluding the value of current record and taking mean
        Train_DS_SUM[feature+'_mean'] =  (Train_DS_SUM[feature+'_sum'] - Train_DS_SUM['Hazard'])/(Train_DS_SUM[feature+'_count']- 1)
        #Train_DS_SUM[feature+'_mean'] =  (Train_DS_SUM[feature+'_sum'])/(Train_DS_SUM[feature+'_count'])
        Train_DS_SUM = Train_DS_SUM.drop([feature+'_sum',feature+'_count'], axis = 1)

        #Train_DS_SUM  = Train_DS_SUM.merge(Feature_median, on=feature)
        #Train_DS_SUM  = Train_DS_SUM.merge(Feature_SD, on=feature)
        #Train_DS_SUM  = Train_DS_SUM.merge(Feature_mode, on=feature)

        Actual_DS_SUM = Actual_DS_SUM.merge(Feature_mean, on=feature)
        # taking the mean of each category
        Actual_DS_SUM[feature+'_mean'] =  (Actual_DS_SUM[feature+'_sum'])/(Actual_DS_SUM[feature+'_count'])
        Actual_DS_SUM = Actual_DS_SUM.drop([feature+'_sum',feature+'_count'], axis = 1)

        #Actual_DS_SUM = Actual_DS_SUM.merge(Feature_median, on=feature)
        #Actual_DS_SUM = Actual_DS_SUM.merge(Feature_SD, on=feature)
        #Actual_DS_SUM = Actual_DS_SUM.merge(Feature_mode, on=feature)

    Train_DS_SUM   = Train_DS_SUM.sort(['Id'], ascending=[True]).reset_index(drop=True).drop(['Id'], axis=1)
    Actual_DS_SUM  = Actual_DS_SUM.sort(['Id'], ascending=[True]).reset_index(drop=True).drop(['Id'], axis=1)

    Train_DS_SUM   = Train_DS_SUM.ix[:,'T1_V4_mean':].fillna(0)
    Actual_DS_SUM  = Actual_DS_SUM.ix[:,'T1_V4_mean':].fillna(0)

    ####################################################################################################################

    y = Train_DS.Hazard.values

    Train_DS = Train_DS.drop(['Hazard'], axis = 1)

    Train_DS  = Train_DS.drop(['T2_V10','T2_V7','T1_V13','T1_V10'], axis = 1)
    Actual_DS = Actual_DS.drop(['T2_V10','T2_V7','T1_V13','T1_V10'], axis = 1)

    # global columns
    columns = Train_DS.columns
    col_types = (Train_DS.dtypes).reset_index(drop=True)

    ####################################################################################################################
    #perform De-vectorizer
    # Train_Dict_DS = Train_DS.T.to_dict().values()
    # Actual_Dict_DS = Actual_DS.T.to_dict().values()
    #
    # vec = DictVectorizer(sparse=False)
    # Train_Dict_DS = vec.fit_transform(Train_Dict_DS)
    # Actual_Dict_DS = vec.transform(Actual_Dict_DS)

    vec = DictVectorizer(sparse=False)
    x_dv = vec.fit_transform((Train_DS.append(Actual_DS)).reset_index(drop=True).T.to_dict().values())
    Train_Dict_DS = x_dv[:len(Train_DS), :]
    Actual_Dict_DS = x_dv[len(Train_DS):, :]

    # for i in range(Train_DS.shape[1]):
    #     if col_types[i]  != 'object':
    #        Train_Dict_DS = np.delete(Train_Dict_DS, i, 1)
    #        Actual_Dict_DS = np.delete(Actual_Dict_DS, i, 1)

    Train_DS = np.array(Train_DS)
    Actual_DS = np.array(Actual_DS)

    ####################################################################################################################

    print("Starting label encoding")
    # label encode the categorical variables
    for i in range(Train_DS.shape[1]):
        if col_types[i] =='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(Train_DS[:,i]) + list(Actual_DS[:,i]))
            Train_DS[:,i] = lbl.transform(Train_DS[:,i])
            Actual_DS[:,i] = lbl.transform(Actual_DS[:,i])

    #rjklllllllllllllllllllllllllllllllllllllllllllllllllle remove it
    Train_DS = Train_Dict_DS
    Actual_DS = Actual_Dict_DS

    ####################################################################################################################
    #Get some new features
    Train_DS = np.append(Train_DS, np.amax(Train_DS,axis=1).reshape(-1,1),1)
    Train_DS = np.append(Train_DS, np.sum(Train_DS,axis=1).reshape(-1,1),1)

    Actual_DS = np.append(Actual_DS, np.amax(Actual_DS,axis=1).reshape(-1,1),1)
    Actual_DS = np.append(Actual_DS, np.sum(Actual_DS,axis=1).reshape(-1,1),1)

    ####################################################################################################################

    #Merge De-vectorizer
    #Train_DS = np.append(Train_DS,Train_Dict_DS,1)
    #Actual_DS = np.append(Actual_DS,Actual_Dict_DS,1)

    Train_DS = np.append(Train_DS,Train_DS_SUM,1)
    Actual_DS = np.append(Actual_DS,Actual_DS_SUM,1)

    Train_DS = Train_DS.astype(float)
    Actual_DS = Actual_DS.astype(float)

    # print("starting TFID conversion...")
    # tfv = TfidfTransformer()
    # tfv.fit(Train_DS)
    # Train_DS1 =  tfv.transform(Train_DS).toarray()
    # Actual_DS1 = tfv.transform(Actual_DS).toarray()
    #
    # Train_DS = np.append(Train_DS,Train_DS1,1)
    # Actual_DS = np.append(Actual_DS,Actual_DS1,1)

    print("Starting log transformation")
    # Train_DS_log2 = np.log(2**Train_DS)/np.log(2)
    # Train_DS_log3 = np.log(3**Train_DS)/np.log(3)
    # Train_DS_log4 = np.log(4**Train_DS)/np.log(4)
    # Train_DS_log5 = np.log(5**Train_DS)/np.log(5)
    # Train_DS_log10 = np.log(10**Train_DS)/np.log(10)
    # Train_DS_log12 = np.log(12**Train_DS)/np.log(12)
    # Train_DS = np.concatenate((Train_DS, Train_DS_log2, Train_DS_log3, Train_DS_log4, Train_DS_log5, Train_DS_log10, Train_DS_log12),axis=1)
    # Train_DS = np.concatenate((Train_DS, Train_DS_log2, Train_DS_log3),axis=1)

    # Actual_DS_log2 = np.log(2**Actual_DS)/np.log(2)
    # Actual_DS_log3 = np.log(3**Actual_DS)/np.log(3)
    # Actual_DS_log4 = np.log(4**Actual_DS)/np.log(4)
    # Actual_DS_log5 = np.log(5**Actual_DS)/np.log(5)
    # Actual_DS_log10 = np.log(10**Actual_DS)/np.log(10)
    # Actual_DS_log12 = np.log(12**Actual_DS)/np.log(12)
    # Actual_DS = np.concatenate((Actual_DS,Actual_DS_log2,Actual_DS_log3,Actual_DS_log4,Actual_DS_log5,Actual_DS_log10,Actual_DS_log12),axis=1)
    # Actual_DS = np.concatenate((Actual_DS, Actual_DS_log2, Actual_DS_log3),axis=1)

    Train_DS, y = shuffle(Train_DS, y, random_state=21)

    Train_DS = np.log( 1 + Train_DS)
    Actual_DS = np.log( 1 + Actual_DS)

    #Setting Standard scaler for data
    stdScaler = StandardScaler()
    stdScaler.fit(Train_DS,y)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    print("***************Ending Data cleansing***************")

    return Train_DS, Actual_DS, y

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def RFR_Regressor(Train_DS, y, Actual_DS, Sample_DS, Parms_DS_RF, Grid, Ensemble):

    print("***************Starting Random Forest Regressor***************")
    t0 = time()
    n_iter_search = 500

    # Train_DS = np.log( 1 + Train_DS)
    # Actual_DS = np.log( 1 + Actual_DS)
    #
    # #Setting Standard scaler for data
    # stdScaler = StandardScaler()
    # stdScaler.fit(Train_DS,y)
    # Train_DS = stdScaler.transform(Train_DS)
    # Actual_DS = stdScaler.transform(Actual_DS)

    Train_DS, y = shuffle(Train_DS, y, random_state=21)

    if Grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "max_depth": [1, 3, 5,8,10,12,15,20,25, None],
                      "max_features": sp_randint(1, 27),
                      "min_samples_split": sp_randint(1, 27),
                      "min_samples_leaf": sp_randint(1, 27),
                      "bootstrap": [True, False]
                     }

        clf = RandomForestRegressor(n_estimators=200)

        # run randomized search
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = gini_scorer,cv=10,n_jobs=-1)

        start = time()
        clf.fit(Train_DS, y)

        print("RandomizedSearchCV took %.2f seconds for %d candidates"
                " parameter settings." % ((time() - start), n_iter_search))
        Parms_DS_Out = report(clf.grid_scores_,n_top=n_iter_search)
        Parms_DS_Out.to_csv(file_path+'Parms_DS_RF2.csv')

        Parms_DS_RF = Parms_DS_Out
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        # print(clf.grid_scores_)
        # print(clf.best_score_)
        # print(clf.best_params_)
        # print(clf.scorer_)

        #Predict actual model
        pred_Actual = clf.predict(Actual_DS)
        print("Actual Model predicted")

        #Get the predictions for actual data set
        preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
        preds.to_csv(file_path+'output/Submission_Roshan_RF_1.csv', index_label='Id')

    if Ensemble:

        print("Starting ensembling")
        Ensemble_DS = pd.DataFrame()

        for i in range(20):
            scores=[]
            if (np.isnan(Parms_DS_RF['max_depth'][i])):
                max_depth_val = None
            else:
                max_depth_val = int(Parms_DS_RF['max_depth'][i])

            clf = RandomForestRegressor(n_estimators=2000
                                       ,min_samples_leaf=int(Parms_DS_RF['min_samples_leaf'][i])
                                       ,max_features=int(Parms_DS_RF['max_features'][i])
                                       ,bootstrap=Parms_DS_RF['bootstrap'][i]
                                       ,min_samples_split=int(Parms_DS_RF['min_samples_split'][i])
                                       ,max_depth=max_depth_val
                                       ,n_jobs=-1)

            clf.fit(Train_DS, y)

            Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
            # scores.append(Nfold_score)
            # print(" %d-iteration... %s " % (i+1,scores))

            pred_Actual = clf.predict(Actual_DS)
            Ensemble_DS[i] = pred_Actual
            print(" %d - Model Completed..." % (i+1))

        Ensemble_DS.to_csv(file_path+'Ensemble_DS_RF2.csv')

    if Grid == False and Ensemble==False:

        #CV:0.3705
        print("Starting normal model prediction")
        # clf = RandomForestRegressor(n_estimators=500,min_samples_leaf=13,max_features=13,bootstrap=True,
        #                             min_samples_split=13,max_depth=None)

        #CV:371411 (0.3680 wtih divect),0.3749 with Label enc and Devect in 500,
        #  .3722 in 1000 , .3732 IN 2000, .3732 IN 3000 (so 2K is fine)
        clf = RandomForestRegressor(n_estimators=500,min_samples_leaf=18,max_features=13,bootstrap=True,
                                    min_samples_split=23,max_depth=25)

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        clf.fit(Train_DS, y)

        # feature = pd.DataFrame()
        # feature['imp'] = clf.feature_importances_
        # feature['col'] = columns
        # feature = feature.sort(['imp'], ascending=False)
        # print(feature)

        #Predict actual model
        pred_Actual = clf.predict(Actual_DS)
        print("Actual Model predicted")

        #Get the predictions for actual data set
        preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
        preds.to_csv(file_path+'output/Submission_Roshan_RF_2.csv', index_label='Id')

    print("***************Ending Random Forest Regressor***************")
    return pred_Actual

########################################################################################################################
#XGB Regressor
########################################################################################################################
def XGB_Regressor(Train_DS, y, Actual_DS, Sample_DS, Parms_DS_XGB, Grid, Ensemble):

    print("***************Starting xgb Regressor (sklearn)***************")
    t0 = time()

    n_iter_search = 500

    Train_DS, y = shuffle(Train_DS, y, random_state=21)

    if Grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "n_estimators": [10],
                      "max_depth": sp_randint(1, 25),
                      "min_child_weight": sp_randint(1, 25),
                      "subsample": [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9, 1],
                      "colsample_bytree": [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9, 1],
                      "silent": [True],
                      "gamma": [0.5, 0.6,0.7,0.8,0.9, 1,2]
                     }

        clf = xgb.XGBRegressor(nthread=4)

        # run randomized search

        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = gini_scorer,cv=10)

        start = time()
        clf.fit(Train_DS, y)

        print("RandomizedSearchCV took %.2f seconds for %d candidates"
                " parameter settings." % ((time() - start), n_iter_search))

        Parms_DS_Out = report(clf.grid_scores_,n_top=n_iter_search)
        Parms_DS_Out.to_csv(file_path+'Parms_DS_XGB_1001.csv')

        Parms_DS_XGB = Parms_DS_Out

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

        #Predict actual model
        pred_Actual = clf.predict(Actual_DS)
        print("Actual Model predicted")

        #Get the predictions for actual data set
        preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
        preds.to_csv(file_path+'output/Submission_Roshan_XGB_1.csv', index_label='Id')

    if Ensemble:

        print("Starting ensembling")
        Ensemble_DS = pd.DataFrame()

        for i in range(20):
            scores=[]
            clf = xgb.XGBRegressor(n_estimators     = 2000
                                  ,max_depth        = Parms_DS_XGB['max_depth'][i]
                                  ,learning_rate    = 0.01
                                  ,nthread          = 4
                                  ,min_child_weight = Parms_DS_XGB['min_child_weight'][i]
                                  ,subsample        = Parms_DS_XGB['subsample'][i]
                                  ,colsample_bytree = Parms_DS_XGB['colsample_bytree'][i]
                                  ,silent           = True
                                  ,gamma            = Parms_DS_XGB['gamma'][i])

            clf.fit(Train_DS, y)

            Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
            #scores.append(Nfold_score)
            #print(" %d-iteration... %s " % (i+1,scores))

            pred_Actual = clf.predict(Actual_DS)
            Ensemble_DS[i] = pred_Actual
            print(" %d - Model Completed..." % (i+1))

        Ensemble_DS.to_csv(file_path+'Ensemble_DS_XGB_1.csv')

    if Grid == False and Ensemble==False:
        #CV:0.38604935169439381, LB:0.382479
        #CV:0.38614992702270973 (with std scaler)
        # clf = xgb.XGBRegressor(n_estimators=1000,max_depth=7,learning_rate=0.01,nthread=2,min_child_weight=5,
        #                         subsample=0.8,colsample_bytree=0.8,silent=True,gamma=1)

        #CV:0.0.38540501304758473
        # clf = xgb.XGBRegressor(n_estimators=1000,max_depth=8,learning_rate=0.01,nthread=4,min_child_weight=5,
        #                         subsample=0.8,colsample_bytree=0.8,silent=True,gamma=1)

        #CV:0.38672594800194787
        clf = xgb.XGBRegressor(n_estimators=2000,max_depth=6,learning_rate=0.01,nthread=4,min_child_weight=15,
                                subsample=1,colsample_bytree=0.5,silent=True,gamma=0.8)

        # CV : 0.38594904255042506)
        # clf = xgb.XGBRegressor(n_estimators=1000,max_depth=5,learning_rate=0.02,nthread=4,min_child_weight=1,
        #                         subsample=1,colsample_bytree=0.9,silent=True,gamma=1)
        #
        # CV :  0.38335661759105549 , 0.3877 in 2000 iter
        # clf = xgb.XGBRegressor(n_estimators=2000,max_depth=5,learning_rate=0.01,nthread=4,min_child_weight=19,
        #                         subsample=1,colsample_bytree=0.3,silent=True,gamma=0.6)

        # CV :  0.3850 in 1000 , 0.3877 in 2000 iter
        clf = xgb.XGBRegressor(n_estimators=2000,max_depth=5,learning_rate=0.01,nthread=4,min_child_weight=20,
                                subsample=0.8,colsample_bytree=0.4,silent=True,gamma=0.6)

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        clf.fit(Train_DS, y)

        #Predict actual model
        pred_Actual = clf.predict(Actual_DS)
        print("Actual Model predicted")

        #Get the predictions for actual data set
        preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
        preds.to_csv(file_path+'output/Submission_Roshan_XGB_1.csv', index_label='Id')

    print("***************Ending xgb Regressor (sklearn)***************")
    return pred_Actual

########################################################################################################################
#Neural network Regressor
########################################################################################################################
def NN1_Regressor(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting NN Regressor ***************")

    t0 = time()

    Train_DS = np.log( 1 + Train_DS)
    Actual_DS = np.log( 1 + Actual_DS)

    #Setting Standard scaler for data
    stdScaler = StandardScaler()
    stdScaler.fit(Train_DS,y)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

    if grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

    else:

        y = y.reshape((-1, 1))

        Actual_DS  = Actual_DS.astype('float32')
        Train_DS   = Train_DS.astype('float32')

        y       = y.astype('float32')

        Train_DS, y = shuffle(Train_DS, y, random_state=42)

        print("Starting NN without grid")

        #cv=0.28 , LB=0.2718
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

        output_nonlinearity=identity,
        output_num_units=1,

        #optimization method
        #update=sgd,
        update=nesterov_momentum,
        #update=adagrad,
        update_learning_rate=0.001,
        update_momentum=0.9,

        eval_size = 0.2,
        regression=True,
        max_epochs=75,
        verbose=1
        )

        clf.fit(Train_DS,y)

        _, X_valid, _, y_valid = clf.train_test_split(Train_DS, y, clf.eval_size)

        y_pred=clf.predict(X_valid)
        score=normalized_gini(y_valid, y_pred)
        print("Best score: %0.3f" % score)

    pred_Actual = clf.predict(Actual_DS)
    print("Actual Model predicted")

    #Get the predictions for actual data set
    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_NN_1.csv', index_label='Id')

    print("***************Ending NN Regressor ***************")
    return pred_Actual

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def Misc_Regressor(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting Misc Regressor **********************")
    t0 = time()

    # Train_DS = np.log( 1 + Train_DS)
    # Actual_DS = np.log( 1 + Actual_DS)
    #
    # #Setting Standard scaler for data
    # stdScaler = StandardScaler()
    # stdScaler.fit(Train_DS,y)
    # Train_DS = stdScaler.transform(Train_DS)
    # Actual_DS = stdScaler.transform(Actual_DS)

    # pca = PCA(n_components=200)
    # pca.fit(Train_DS,y)
    # Train_DS = pca.transform(Train_DS)
    # Actual_DS = pca.transform(Actual_DS)

    if grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        # param_dist = {
        #               "kernel": ['rbf'],
        #               "C": [1,10,100,0.1,0.01,0.05,0.5 ],
        #               "gamma": [0,1,10,0.1,0.01,0.001 ]
        #              }
        #
        # clf = SVR(max_iter=-1)
        #
        # clf = GridSearchCV(estimator = clf, param_grid=param_dist, scoring=gini_scorer,
        #                              verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)


        ################################################################################################################
        # specify parameters and distributions to sample from
        n_iter_search = 100
        param_dist = {
                      "metric": ['minkowski','euclidean','manhattan','chebyshev','wminkowski','seuclidean','mahalanobis',
                                 'haversine','hamming','canberra','braycurtis'],
                      "n_neighbors": [5,10,15,20,25,50,100,150,200,250,300 ]
                     }

        clf = KNeighborsRegressor()

        clf = GridSearchCV(estimator = clf, param_grid=param_dist, scoring=gini_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)
        ################################################################################################################

        # run randomized search
        # n_iter_search = 100
        # clf = RandomizedSearchCV(clf, param_distributions=param_dist,
        #                                    n_iter=n_iter_search, scoring = gini_scorer,cv=10)

        start = time()
        clf.fit(Train_DS, y)

        Parms_DS_Out = report(clf.grid_scores_,n_top=n_iter_search)
        Parms_DS_Out.to_csv(file_path+'Parms_DS_Misc_1001.csv')

        # print("RandomizedSearchCV took %.2f seconds for %d candidates"
        #         " parameter settings." % ((time() - start), n_iter_search))
        report(clf.grid_scores_)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        print(clf.grid_scores_)
        print(clf.best_score_)
        print(clf.best_params_)
        print(clf.scorer_)
    else:
        #CV:0.24xxxxxxx
        clf = KNeighborsRegressor(n_neighbors=20,metric='euclidean')

        #clf = RadiusNeighborsRegressor()

        #CV:0.31
        #clf = SVR(kernel='rbf',max_iter=-1)

        #CV:0.335511976443 , including normal, Devect and TFIDF , LB:0.329886
        #clf = ElasticNet(alpha=0.1,  l1_ratio=0.1)

        #CV:Error...array is too big
        #clf = ARDRegression()

        #0.015
        #clf = Lars()

        #CV:0.29
        #clf = AdaBoostRegressor()

        #CV:0.331888
        #clf = Lasso(alpha=0.02)

        #CV: 0.336680274991
        #clf = Ridge(alpha=0.02)

        #cv:0.3375 with log transform
        #clf = BayesianRidge()

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict(Actual_DS)
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_Misc_1.csv', index_label='Id')

    print("***************Ending Misc Regressor **********************")
    return pred_Actual

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def XGO_Regressor(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting xgb Regressor (original)***************")
    t0 = time()

    if grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "n_estimators": [100],
                      "max_depth": sp_randint(1, 11),
                      "min_child_weight": sp_randint(1, 11),
                      "subsample": [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9, 1],
                      "colsample_bytree": [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9, 1],
                      "silent": [True],
                      "gamma": [0.5, 0.6,0.7,0.8,0.9, 1]
                     }

        clf = xgb.XGBRegressor()

        # run randomized search
        n_iter_search = 1000
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = gini_scorer,cv=10,n_jobs=-1)

        start = time()
        clf.fit(Train_DS, y)

        print("RandomizedSearchCV took %.2f seconds for %d candidates"
                " parameter settings." % ((time() - start), n_iter_search))
        report(clf.grid_scores_)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        print(clf.grid_scores_)
        print(clf.best_score_)
        print(clf.best_params_)
        print(clf.scorer_)
    else:

        #CV: 0.382764 (best as of now)
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.01
        params["min_child_weight"] = 5
        params["subsample"] = 0.8
        params["colsample_bytree"] = 0.8
        params["scale_pos_weight"] = 1.0
        params["silent"] = 1
        params["max_depth"] = 7

        plst = list(params.items())

        #Using 5000 rows for early stopping.
        offset = 5000

        num_rounds = 2000
        xgtest = xgb.DMatrix(Actual_DS)

        #create a train and validation dmatrices
        xgtrain = xgb.DMatrix(Train_DS[offset:,:], label=y[offset:])
        xgval = xgb.DMatrix(Train_DS[:offset,:], label=y[:offset])

        #train using early stopping and predict
        watchlist = [(xgtrain, 'train'),(xgval, 'val')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
        preds1 = model.predict(xgtest)

        #reverse train and labels and use different 5k for early stopping.
        # this adds very little to the score but it is an option if you are concerned about using all the data.
        train = Train_DS[::-1,:]
        labels = np.log(y[::-1])

        xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
        xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

        watchlist = [(xgtrain, 'train'),(xgval, 'val')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
        preds2 = model.predict(xgtest)

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
        preds3 = model.predict(xgtest)

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
        preds4 = model.predict(xgtest)

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
        preds5 = model.predict(xgtest)

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
        preds6 = model.predict(xgtest)

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
        preds7 = model.predict(xgtest)

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
        preds8 = model.predict(xgtest)


        #combine predictions
        #since the metric only cares about relative rank we don't need to average
        pred_Actual = preds1 + preds2+preds3+preds4+preds5+preds6+preds7+preds8

        print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_XGB_orig_1.csv', index_label='Id')

    print("***************Ending xgb Regressor (original)***************")
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, gini_scorer

    # Normalized Gini Scorer
    gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)

    if(platform.system() == "Windows"):
        file_path = 'C:/Python/Others/data/Kaggle/Liberty_Mutual_Group/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Liberty_Mutual_Group/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS        = pd.read_csv(file_path+'train.csv',sep=',', index_col=0)
    Actual_DS       = pd.read_csv(file_path+'test.csv',sep=',', index_col=0)
    Sample_DS = pd.read_csv(file_path+'sample_submission.csv',sep=',')
    Parms_XGB_DS = pd.read_csv(file_path+'Parms_DS_XGB_1001.csv',sep=',')
    Parms_RF_DS = pd.read_csv(file_path+'Parms_DS_RF2.csv',sep=',')

    Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS)

    pred_Actual = RFR_Regressor(Train_DS, y, Actual_DS, Sample_DS, Parms_RF_DS , Grid=False , Ensemble= False)
    #pred_Actual = XGB_Regressor(Train_DS, y, Actual_DS, Sample_DS, Parms_XGB_DS, Grid=True , Ensemble= True)

    #pred_Actual = XGO_Regressor(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = Misc_Regressor(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = NN1_Regressor(Train_DS, y, Actual_DS, Sample_DS, grid=False)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)
