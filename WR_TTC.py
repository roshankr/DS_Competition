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

    print("***************Starting Kfold Cross validation***************")

    X =np.array(X)
    scores=[]

    ss = StratifiedShuffleSplit(y, n_iter=5,test_size=0.2, random_state=21, indices=None)
    #ss = KFold(len(y), n_folds=5,shuffle=False,indices=None)

    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

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
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid_New(X, y, clf):

    print("***************Starting Kfold Cross validation***************")

    X =np.array(X)
    scores=[]

    unique_labels = np.unique(X[:,0])

    X = pd.DataFrame(X)
    X['y'] = y

    #ss = StratifiedShuffleSplit(y, n_iter=5,test_size=0.2, random_state=21, indices=None)
    #ss = KFold(len(y), n_folds=5,shuffle=False,indices=None)
    #ss = StratifiedKFold(y, n_folds=5,shuffle=False,indices=unique_labels)

    i = 1

    for i in range(5):
        test_labels = np.random.choice(unique_labels, size=len(unique_labels)*0.2,replace=True)

        X_test   = X[X[0].isin(test_labels)]
        y_test   = X_test['y']
        X_test   = X_test.drop(['y',0], axis = 1)

        X_train  = X[~X[0].isin(test_labels)]
        y_train  = X_train['y']
        X_train  = X_train.drop(['y',0], axis = 1)

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

    # test = Train_DS.iloc[np.random.permutation(len(Train_DS))]
    # test = test.head(100000).sort(columns='VisitNumber',ascending=True)
    # pd.DataFrame(test).to_csv(file_path+'train_100000.csv')
    #
    # test = Actual_DS.iloc[np.random.permutation(len(Actual_DS))]
    # test = test.head(100000).sort(columns='VisitNumber',ascending=True)
    # pd.DataFrame(test).to_csv(file_path+'test_100000.csv')

    y = Train_DS.TripType.values
    Train_DS = Train_DS.drop(['TripType'], axis = 1)
    ##----------------------------------------------------------------------------------------------------------------##
    #Label Encode DepartmentDescription
    lbl = preprocessing.LabelEncoder()
    lbl.fit((list(Train_DS['Weekday'].astype(str)) + list(Actual_DS['Weekday'].astype(str))))
    Train_DS['Weekday'] = lbl.transform(Train_DS['Weekday'].astype(str))
    Actual_DS['Weekday'] = lbl.transform(Actual_DS['Weekday'].astype(str))

    # #weekday one hot encoding
    # print("weekday one hot encoding")
    # New_DS = pd.concat([Train_DS, Actual_DS])
    # dummies = pd.get_dummies(New_DS['Weekday'])
    # cols_new = [ 'Weekday'+"_"+str(s) for s in list(dummies.columns)]
    # New_DS[cols_new] = dummies
    #
    # Train_DS = New_DS.head(len(Train_DS))
    # Actual_DS = New_DS.tail(len(Actual_DS))
    #
    # Train_DS = Train_DS.drop(['Weekday'], axis = 1)
    # Actual_DS = Actual_DS.drop(['Weekday'], axis = 1)

    ##----------------------------------------------------------------------------------------------------------------##
    #Label Encode DepartmentDescription
    lbl = preprocessing.LabelEncoder()
    lbl.fit((list(Train_DS['DepartmentDescription'].astype(str)) + list(Actual_DS['DepartmentDescription'].astype(str))))
    Train_DS['DepartmentDescription'] = lbl.transform(Train_DS['DepartmentDescription'].astype(str))
    Actual_DS['DepartmentDescription'] = lbl.transform(Actual_DS['DepartmentDescription'].astype(str))

    ##----------------------------------------------------------------------------------------------------------------##
    print("Get the total number of items")

    Train_DS = Train_DS.reset_index()
    Actual_DS = Actual_DS.reset_index()

    #Get the total number of items
    Total_Items = Train_DS.groupby(['VisitNumber']).agg({'ScanCount': [np.sum]}).reset_index()
    Total_Items.columns = ['VisitNumber','total_items']
    Train_DS = Train_DS.merge(Total_Items, on='VisitNumber', how =  'left')

    Total_Items = Actual_DS.groupby(['VisitNumber']).agg({'ScanCount': [np.sum]}).reset_index()
    Total_Items.columns = ['VisitNumber','total_items']
    Actual_DS = Actual_DS.merge(Total_Items, on='VisitNumber', how =  'left')

    print("Get the total different unique items (FinelineNumber)")

    #Get the total different unique items (FinelineNumber)
    Total_Items = Train_DS.groupby(['VisitNumber']).FinelineNumber.nunique().reset_index()
    Total_Items.columns = ['VisitNumber','total_unique_items']
    Train_DS = Train_DS.merge(Total_Items, on='VisitNumber', how =  'left')

    Total_Items = Actual_DS.groupby(['VisitNumber']).FinelineNumber.nunique().reset_index()
    Total_Items.columns = ['VisitNumber','total_unique_items']
    Actual_DS = Actual_DS.merge(Total_Items, on='VisitNumber', how =  'left')

    print("Get the total different unique depts (DepartmentDescription)")

    #Get the total different unique depts (DepartmentDescription)
    Total_Items = Train_DS.groupby(['VisitNumber']).DepartmentDescription.nunique().reset_index()
    Total_Items.columns = ['VisitNumber','total_unique_depts']
    Train_DS = Train_DS.merge(Total_Items, on='VisitNumber', how =  'left')

    Total_Items = Actual_DS.groupby(['VisitNumber']).DepartmentDescription.nunique().reset_index()
    Total_Items.columns = ['VisitNumber','total_unique_depts']
    Actual_DS = Actual_DS.merge(Total_Items, on='VisitNumber', how =  'left')

    #Maintain the order after merge
    Train_DS   = Train_DS.sort(['index'], ascending=[True]).reset_index(drop=True).drop(['index'], axis=1)
    Actual_DS   = Actual_DS.sort(['index'], ascending=[True]).reset_index(drop=True).drop(['index'], axis=1)

    ##----------------------------------------------------------------------------------------------------------------##
    #Any scaling , log transformations....

    print("Any scaling , log transformations")

    Train_DS = Train_DS.replace([np.inf, -np.inf], np.nan)
    Actual_DS = Actual_DS.replace([np.inf, -np.inf], np.nan)

    Train_DS = Train_DS.fillna(0)
    Actual_DS = Train_DS.fillna(0)

    # Train_DS = np.log(1+ Train_DS)
    # Actual_DS = np.log(1+ Actual_DS)

    #Setting Standard scaler for data
    stdScaler = StandardScaler(with_mean=True, with_std=True)
    stdScaler.fit(Train_DS,y)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

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

        #
        clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_split=1,max_features='auto',bootstrap=True,
                                        max_depth = 8, min_samples_leaf = 4,oob_score=True,criterion='entropy')

        clf = RandomForestClassifier(n_jobs=-1, n_estimators=500)

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        sys.exit(0)

        #clf = RandomForestClassifier(n_jobs=-1, n_estimators=2000)
        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        clf.fit(Train_DS, y)

        #
        feature = pd.DataFrame()
        feature['imp'] = clf.feature_importances_
        feature['col'] = Train_DS1.columns
        feature = feature.sort(['imp'], ascending=False).reset_index(drop=True)
        print(feature)
        pd.DataFrame(feature).to_csv(file_path+'feature_imp.csv')

        sys.exit(0)
    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_RFC_filter_2.csv', index_label='ID')

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

        param_grid = {'n_estimators': [100],
                      'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                      'min_child_weight': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                      'subsample': [0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9,1],
                      'colsample_bytree': [0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9,1],
                      'silent':[True],
                      'gamma':[2,1,0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9]
                     }

        # clf = GridSearchCV(xgb.XGBClassifier(),param_grid, scoring='roc_auc',
        #                    verbose=1,cv=10)

        #run randomized search
        n_iter_search = 3000
        clf = xgb.XGBClassifier(nthread=-1)
        clf = RandomizedSearchCV(clf, param_distributions=param_grid,
                                           n_iter=n_iter_search, scoring = 'roc_auc',cv=10)

        start = time()
        clf.fit(Train_DS, y)

        print("GridSearchCV completed")
        report(clf.grid_scores_)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

    else:

        #from Kaggle
        clf = xgb.XGBClassifier(n_estimators=500,max_depth=9,learning_rate=0.01,nthread=2,min_child_weight=6,
                             subsample=0.7,colsample_bytree=0.5,silent=True, gamma = 4)

        clf = xgb.XGBClassifier(n_estimators=100,nthread=-1)

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        sys.exit(0)
        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set

    preds = pd.DataFrame(pred_Actual, index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_xgb_filter_2.csv', index_label='ID')

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
        clf = LogisticRegression()

        #print("Adaboost")
        #CV: 0.7099
        #clf = AdaBoostClassifier(n_estimators=100)

        # print("BaggingClassifier")
        # #CV:
        # clf = BaggingClassifier(n_estimators=100)
        # Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        #
        print("ExtraTreesClassifier")
        #CV:0.7247
        clf = ExtraTreesClassifier(n_estimators=100)
        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        print("MultinomialNB")
        #CV:
        clf = MultinomialNB()
        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        print("BernoulliNB")
        #CV:
        clf = BernoulliNB()
        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        sys.exit(0)

        clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')
        clf.fit(Train_DS, y)

        # feature = pd.DataFrame()
        # feature['imp'] = clf.feature_importances_
        # feature['col'] = Train_DS1.columns
        # feature = feature.sort(['imp'], ascending=False).reset_index(drop=True)
        # print(feature)


    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_Misc_filter_2.csv', index_label='ID')

    print("***************Ending Random Forest Classifier***************")
    return pred_Actual
########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, Train_DS1, Featimp_DS

    #random.seed(1)

    if(platform.system() == "Windows"):

        file_path = 'C:/Python/Others/data/Kaggle/Walmart_Recruiting_TTC/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Walmart_Recruiting_TTC/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS      = pd.read_csv(file_path+'train_Grouped.csv',sep=',')
    Actual_DS     = pd.read_csv(file_path+'test_Grouped.csv',sep=',')
    Sample_DS     = pd.read_csv(file_path+'sample_submission.csv',sep=',')

    #For testing only
    #Train_DS      = pd.read_csv(file_path+'train_100000.csv',sep=',', index_col=0,nrows = 100000 ).reset_index(drop=True)
    #Actual_DS     = pd.read_csv(file_path+'test_100000.csv',sep=',', index_col=0,nrows = 100000).reset_index(drop=True)

    Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS)

    pred_Actual = XGB_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    #pred_Actual  = RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    #pred_Actual  = Misc_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)