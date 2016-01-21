import requests
import numpy as np
import scipy as sp
import sys
import platform
import pandas as pd
from time import time
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
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
#import xgboost as xgb
########################################################################################################################
#Springleaf Marketing Response
########################################################################################################################
#--------------------------------------------Algorithm : Random Forest :------------------------------------------------
#Random Forest :
#--------------------------------------------Algorithm : XGB------------------------------------------------------------
#XGB :

#--------------------------------------------Suggestions, Ideas---------------------------------------------------------
#Suggestions, Ideas

# Best score 6/10/2015:
#XGB : 0.782434654905
#RFC : 0.767411324977
#LOG : 0.760828849164

#--------------------------------------------with only 7K records-------------------------------------------------------
# RF : 0.7410 - 7414 (with 7k)

#With stratified fold
#RF: 0.7366 - 0.7370
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
        y_pred=clf.predict_proba(X_test)[:,1]
        scores.append(roc_auc_score(y_test,y_pred))
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
def Data_Munging(Train_DS,Actual_DS, Filter_DS):

    print("***************Starting Data cleansing***************")

    global  Train_DS1

    print(np.shape(Train_DS))
    Train_DS = Train_DS.dropna(axis=1, thresh=100)
    Actual_DS = Actual_DS.dropna(axis=1, thresh=100)

    print(np.shape(Train_DS))

    y = Train_DS.target.values
    Train_DS = Train_DS.drop(['target','ID','VAR_0044'], axis = 1)
    Actual_DS = Actual_DS.drop(['ID','VAR_0044'], axis = 1)

    ####################################################################################################################
    #Get column unique count
    columns = Train_DS.columns
    col_types = (Train_DS.dtypes).reset_index(drop=True)

    #Delete object columns with only one Unique values (numeric unique already removed)
    unique_cols = []
    for j in range(Train_DS.shape[1]):
        #if col_types[j] =='object':
            if (len(Train_DS[columns[j]].value_counts(dropna=True))) <= 1:
                #for testing purpose only
                if (columns[j])!='VAR_0214':
                    unique_cols.append(columns[j])

    Train_DS = Train_DS.drop(unique_cols, axis = 1)
    Actual_DS = Actual_DS.drop(unique_cols, axis = 1)

    print("Unique columns deleted")

    ####################################################################################################################
    # Get numeric feature after removing duplicates
    print("Starting Numeric conversion....")

    #Take all Numeric values
    columns = Train_DS.columns
    col_types = (Train_DS.dtypes).reset_index(drop=True)
    cols_type = pd.DataFrame()
    cols_type['name'] = columns
    cols_type['type'] = col_types
    cols_obj = list(cols_type['name'][(cols_type['type'] == 'int64')  | (cols_type['type'] == 'float64')])

    #-------------------------------------------------------------------------------------------------------------------
    #Take only numeric with best range after clustering
    #cols_grp = list(Filter_DS['feature'])
    #cols_grp = list(set(cols_obj).intersection(cols_grp))
    #-------------------------------------------------------------------------------------------------------------------

    Train_DS_New = Train_DS[cols_obj]
    Actual_DS_New = Actual_DS[cols_obj]

    #Deafult all -ve values to -1
    Train_DS_New[Train_DS_New < 0 ] = -10
    Actual_DS_New[Actual_DS_New < 0 ] = -10

    Train_DS_New  = Train_DS_New.fillna(-999)
    Actual_DS_New = Actual_DS_New.fillna(-999)

    # cols_new = Train_DS_New.columns
    # imp = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)
    # Train_DS_New  = pd.DataFrame(imp.fit_transform(Train_DS_New), columns=cols_new)
    # Actual_DS_New = pd.DataFrame(imp.fit_transform(Actual_DS_New), columns=cols_new)

    #-------------------------------------------------------------------------------------------------------------------
    # Devectorize / one hot encoding fro numeric cols

    # New_DS = pd.concat([Train_DS_New, Actual_DS_New])
    # cat_cols =  list(Filter_DS[(Filter_DS['Min']==0) & (Filter_DS['Max']==99) & (Filter_DS['Uniq'] <= 15)]['feature'])
    #
    # for column in cat_cols:
    #     dummies = pd.get_dummies(New_DS[column])
    #     cols_new = [ column+"_"+str(s) for s in list(dummies.columns)]
    #     New_DS[cols_new] = dummies
    #
    # #New_DS = New_DS.drop(cat_cols, axis = 1)
    # Train_DS_New = New_DS.head(len(Train_DS_New))
    # Actual_DS_New = New_DS.tail(len(Actual_DS_New))

    #-------------------------------------------------------------------------------------------------------------------
    #Get the count of hig cardinality categorical numerical features and apply it as feature

    # New_DS = pd.concat([Train_DS_New, Actual_DS_New])
    # Train_DS_New = Train_DS_New.reset_index()
    # Actual_DS_New = Actual_DS_New.reset_index()
    # #
    # Factors =  list(Filter_DS[(Filter_DS['Min']==0) & (Filter_DS['Max']==99) & (Filter_DS['Uniq'] <= 15)]['feature'])
    #
    # for feature in Factors:
    #     Feature_count = New_DS[feature].value_counts().reset_index()
    #     Feature_count.columns = [feature,feature+'_count']
    #
    #     #Merge with Train and Test
    #     Train_DS_New  = Train_DS_New.merge(Feature_count, on=feature, how='left')
    #     Actual_DS_New  = Actual_DS_New.merge(Feature_count, on=feature, how='left')
    #
    # #Maintain the order after merge
    # Train_DS_New   = Train_DS_New.sort(['index'], ascending=[True]).reset_index(drop=True).drop(['index'], axis=1)
    # Actual_DS_New   = Actual_DS_New.sort(['index'], ascending=[True]).reset_index(drop=True).drop(['index'], axis=1)

    #-------------------------------------------------------------------------------------------------------------------
    # Try to get mean of target score using "leave one out" encoding of numerical categorical vars

    # Train_DS_New['target'] = y
    # Train_DS_New = Train_DS_New.reset_index()
    # Actual_DS_New = Actual_DS_New.reset_index()
    #
    # Factors =  list(Filter_DS[(Filter_DS['Min']==0) & (Filter_DS['Max']==99) & (Filter_DS['Uniq'] <= 15)]['feature'])
    # #Factors = ['VAR_0647']
    #
    # for feature in Factors:
    #     Feature_mean = Train_DS_New.groupby([feature]).agg({'target': ['sum', 'count']}).reset_index()
    #     Feature_mean.columns = [feature,feature+'_sum',feature+'_count']
    #     Train_DS_New  = Train_DS_New.merge(Feature_mean, on=feature, how='left')
    #
    #     # excluding the value of current record and taking mean
    #     Train_DS_New[feature+'_mean'] =  (Train_DS_New[feature+'_sum'] - Train_DS_New['target'])/(Train_DS_New[feature+'_count']- 1)
    #     Train_DS_New = Train_DS_New.drop([feature+'_sum',feature+'_count'], axis = 1).replace([np.inf, -np.inf], np.nan).fillna(0)
    #
    #     Actual_DS_New = Actual_DS_New.merge(Feature_mean, on=feature, how='left')
    #     Actual_DS_New[feature+'_mean'] =  (Actual_DS_New[feature+'_sum'])/(Actual_DS_New[feature+'_count'])
    #     Actual_DS_New = Actual_DS_New.drop([feature+'_sum',feature+'_count'], axis = 1).replace([np.inf, -np.inf], np.nan).fillna(0)
    #
    # Train_DS_New   = Train_DS_New.sort(['index'], ascending=[True]).reset_index(drop=True).drop(['index'], axis=1)
    # Actual_DS_New   = Actual_DS_New.sort(['index'], ascending=[True]).reset_index(drop=True).drop(['index'], axis=1)
    #
    # Train_DS_New = Train_DS_New.drop(['target'], axis = 1)

    #Delete only for testing purpose
    #Train_DS_New = Train_DS_New.drop(Factors, axis = 1)
    #Actual_DS_New = Actual_DS_New.drop(Factors, axis = 1)

    #-------------------------------------------------------------------------------------------------------------------
    # Try imputing 99, 999, 999999.... values

    # imp_val_list = [9999,99999,999999,9999999,999999999,9998]
    # for imp_val in imp_val_list:
    #     imp_cols =  Filter_DS[Filter_DS['Max']==imp_val]['feature']
    #
    #     for i in range(10):
    #         Train_DS_New[imp_cols] = Train_DS_New[imp_cols].replace(to_replace=(imp_val-i), value='NaN')
    #
    #     imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0,copy=False)
    #     Train_DS_New[imp_cols] = imp.fit_transform(Train_DS_New[imp_cols])

    #-------------------------------------------------------------------------------------------------------------------
    # Try replacing 99, 999, 999999.... values with -999 ( a common value)

    imp_val_list = [ 9999,99999,999999,9999999,999999999,9998]

    #best now
    imp_val_list = [99999,999999,9999999,999999999]
    #imp_val_list = [999999,9999999,99999999,999999999,999998,9999998,99999998,999999998,999997,9999997,99999997,999999997]

    for imp_val in imp_val_list:
        imp_cols =  Filter_DS[Filter_DS['Max']==imp_val]['feature']

        for i in range(2):
            Train_DS_New[imp_cols]  = Train_DS_New[imp_cols].replace(to_replace=(imp_val-i), value=-999)
            Actual_DS_New[imp_cols] = Actual_DS_New[imp_cols].replace(to_replace=(imp_val-i), value=-999)

    #-------------------------------------------------------------------------------------------------------------------
    Train_DS_New['VAR_0212'] =  Train_DS_New['VAR_0212'] / 100000
    Actual_DS_New['VAR_0212'] =  Actual_DS_New['VAR_0212'] / 100000

    #-------------------------------------------------------------------------------------------------------------------
    #Try binning the numeric data
    # Max_groups = Filter_DS['Uniq']
    # bin_value = 10
    #
    # row_iterator = Filter_DS.iterrows()
    #
    # for i, row in row_iterator:
    #
    #     if Filter_DS.loc[i, 'Uniq'] > 999:
    #         data = np.array(Train_DS_New[str(Filter_DS.loc[i, 'feature'])])
    #
    #         #bins = np.linspace(data.min(), data.max(), bin_value)
    #         bins = np.linspace(0, 999999999, bin_value)
    #
    #         print(bins)
    #         sys.exit(0)
    #         digitized = np.digitize(data, bins)
    #         bin_means = [data[digitized == i].size for i in range(1, len(bins))]
    #
    #         print(digitized)
    #         print(pd.DataFrame(bin_means))
    #
    #         sys.exit(0)
    #
    # data = np.random.random(100)
    # bins = np.linspace(0, 1, 10)
    # digitized = np.digitize(data, bins)
    # bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
    #
    # print(bins)
    # print(data)
    # print(digitized)
    # sys.exit(0)


    #-------------------------------------------------------------------------------------------------------------------
    # Trying to take the avg of each cluster values
    # Max_groups = Filter_DS['cluster'].max()
    #
    # for i in range(Max_groups+1):
    #     clust = list(Filter_DS[Filter_DS['cluster'] == i ] ['feature'])
    #     clust = list(set(cols_obj).intersection(clust))
    #
    #     if len(clust) > 0:
    #         Train_DS_New[str(i)+'_clust_avg'] = Train_DS_New[clust].mean(axis=1)
    #         Actual_DS_New[str(i)+'_clust_avg'] = Actual_DS_New[clust].mean(axis=1)
    #
    # Train_DS_New = Train_DS_New.drop(cols_grp, axis = 1)
    # Actual_DS_New = Actual_DS_New.drop(cols_grp, axis = 1)

    print("Ending Numeric conversion....")

    print(np.shape(Train_DS_New))
    print(np.shape(Actual_DS_New))

    ####################################################################################################################
    # Verify the non-numeric lists
    columns = Train_DS.columns
    col_types = (Train_DS.dtypes).reset_index(drop=True)

    cols_type = pd.DataFrame()
    cols_type['name'] = columns
    cols_type['type'] = col_types
    cols_obj = list(cols_type['name'][cols_type['type'] == 'object'])

    cols_boolean    = ['VAR_0008','VAR_0009','VAR_0010','VAR_0011','VAR_0012','VAR_0043','VAR_0196','VAR_0226','VAR_0229','VAR_0230','VAR_0232','VAR_0236','VAR_0239']
    cols_date       = ['VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159', 'VAR_0166', 'VAR_0167', 'VAR_0168','VAR_0169', 'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179', 'VAR_0204', 'VAR_0217']
    cols_cat        = ['VAR_0001','VAR_0005','VAR_0216','VAR_0222','VAR_0237','VAR_0274','VAR_0283','VAR_0305','VAR_0325','VAR_0342','VAR_0352','VAR_0353','VAR_0354','VAR_0466']
    cols_others     = ['VAR_0200','VAR_0214','VAR_0404','VAR_0467','VAR_0493','VAR_1934']

    cols_boolean = sorted(list(set(cols_obj).intersection(cols_boolean)))
    cols_cat = sorted(list(set(cols_obj).intersection(cols_cat)))

    ####################################################################################################################
    #For the date fields in Train and Actual

    print("Starting Date conversion....")

    monthDict={'JAN':'01', 'FEB':'02', 'MAR':'03', 'APR':'04', 'MAY':'05', 'JUN':'06', 'JUL':'07', 'AUG':'08', 'SEP':'09', 'OCT':'10', 'NOV':'11', 'DEC':'12'}

    #clean up all date fields, extract DD,MMM,YYYY and combine to a single date for further use
    for datecol in cols_date:

        Train_DS[datecol+'_DD'] = Train_DS[datecol].str[:2].fillna(1).astype(int)
        Train_DS[datecol+'_MM'] = Train_DS[datecol].str[2:5].replace(monthDict).fillna(1).astype(int)
        Train_DS[datecol+'_YY'] = Train_DS[datecol].str[5:7].fillna(1).astype(int).apply(lambda x: x+2000)
        Train_DS_New[datecol]   = pd.to_datetime(Train_DS[datecol+'_YY']*10000+Train_DS[datecol+'_MM']*100+Train_DS[datecol+'_DD'],format='%Y%m%d')

        temp = pd.DatetimeIndex(Train_DS_New[datecol])
        Train_DS_New[datecol+'_MM'] = temp.month
        Train_DS_New[datecol+'_DW'] = temp.dayofweek

        Actual_DS[datecol+'_DD'] = Actual_DS[datecol].str[:2].fillna(1).astype(int)
        Actual_DS[datecol+'_MM'] = Actual_DS[datecol].str[2:5].replace(monthDict).fillna(1).astype(int)
        Actual_DS[datecol+'_YY'] = Actual_DS[datecol].str[5:7].fillna(1).astype(int).apply(lambda x: x+2000)
        Actual_DS_New[datecol]   = pd.to_datetime(Actual_DS[datecol+'_YY']*10000+Actual_DS[datecol+'_MM']*100+Actual_DS[datecol+'_DD'],format='%Y%m%d')

        temp = pd.DatetimeIndex(Actual_DS_New[datecol])
        Actual_DS_New[datecol+'_MM'] = temp.month
        Actual_DS_New[datecol+'_DW'] = temp.dayofweek

    # Get Date Differences
    for index , datecol in enumerate(cols_date):
        j = index + 1
        for i in range(j , len(cols_date)):
            newcol = cols_date[index]+"_"+cols_date[i]
            Train_DS_New[newcol] = (Train_DS_New[cols_date[index]] - Train_DS_New[cols_date[i]]).astype('timedelta64[D]')
            Actual_DS_New[newcol] = (Actual_DS_New[cols_date[index]] - Actual_DS_New[cols_date[i]]).astype('timedelta64[D]')

    Train_DS_New = Train_DS_New.drop(cols_date, axis = 1)
    Actual_DS_New = Actual_DS_New.drop(cols_date, axis = 1)

    print("Ending Date conversion....")

    print(np.shape(Train_DS_New))
    print(np.shape(Actual_DS_New))

    ####################################################################################################################
    #Inspect Boolean data and apply conversion
    print("Starting Boolean conversion....")
    Train_DS_New[cols_boolean]  = Train_DS[cols_boolean].fillna(False)
    Actual_DS_New[cols_boolean] = Actual_DS[cols_boolean].fillna(False)

    #print(Train_DS_New.ix[:,'VAR_0226':])

    print("Ending Boolean conversion....")

    print(np.shape(Train_DS_New))
    print(np.shape(Actual_DS_New))

    ####################################################################################################################
    #Inspect Categorical elements and apply conversion

    print("Starting Categorical conversion....")

    Train_DS_New[cols_cat]  = Train_DS[cols_cat].fillna('00')
    Actual_DS_New[cols_cat] = Actual_DS[cols_cat].fillna('00')

    #-------------------------------------------------------------------------------------------------------------------
    #looks like VAR_0274 is state code , but not matching with VAR_0237 (correct as per zipcode-VAR_0241)
    Train_DS_New = Train_DS_New.drop(['VAR_0274'], axis = 1)
    Actual_DS_New = Actual_DS_New.drop(['VAR_0274'], axis = 1)

    #-------------------------------------------------------------------------------------------------------------------
    #looks like VAR_0283 , VAR_0305 and VAR_0325 are showing similar values. So lets get encoding with same value
    cols_temp = ['VAR_0283','VAR_0305','VAR_0325']
    Train_DS_test_key = np.unique(list(np.unique(Train_DS_New[cols_temp].values))+list(np.unique(Actual_DS_New[cols_temp].values)))
    Train_DS_test_val = list(range(0,len(Train_DS_test_key)))
    dictionary = dict(zip(Train_DS_test_key,Train_DS_test_val))
    Train_DS_New[cols_temp]  = Train_DS_New[cols_temp].replace(dictionary)
    Actual_DS_New[cols_temp] = Actual_DS_New[cols_temp].replace(dictionary)

    #-------------------------------------------------------------------------------------------------------------------
    #looks like VAR_0352 , VAR_0353 and VAR_0354 are showing similar values. So lets get encoding with same value
    cols_temp = ['VAR_0352','VAR_0353','VAR_0354']
    Train_DS_test_key = np.unique(list(np.unique(Train_DS_New[cols_temp].values))+list(np.unique(Actual_DS_New[cols_temp].values)))
    Train_DS_test_val = list(range(0,len(Train_DS_test_key)))
    dictionary = dict(zip(Train_DS_test_key,Train_DS_test_val))
    Train_DS_New[cols_temp]  = Train_DS_New[cols_temp].replace(dictionary)
    Actual_DS_New[cols_temp] = Actual_DS_New[cols_temp].replace(dictionary)

    #-------------------------------------------------------------------------------------------------------------------
    # Devectorize / one hot encoding 'VAR_0001','VAR_0005' columns
    for column in ['VAR_0001','VAR_0005']:
        dummies = pd.get_dummies(Train_DS_New[column])
        Train_DS_New[dummies.columns] = dummies

        dummies = pd.get_dummies(Actual_DS_New[column])
        Actual_DS_New[dummies.columns] = dummies

    Train_DS_New = Train_DS_New.drop(['VAR_0001','VAR_0005'], axis = 1)
    Actual_DS_New = Actual_DS_New.drop(['VAR_0001','VAR_0005'], axis = 1)

    #-------------------------------------------------------------------------------------------------------------------
    #'VAR_0342' has 2 char values. May be splitting 1st and 2nd char  and keeping them sep work???

    Train_DS_New['VAR_0342'] = Train_DS_New['VAR_0342'].replace(to_replace='-1', value='01')
    Train_DS_New['VAR_0342_1'] = Train_DS_New['VAR_0342'].str[:1]
    Train_DS_New['VAR_0342_2'] = Train_DS_New['VAR_0342'].str[1:2]

    Actual_DS_New['VAR_0342'] = Actual_DS_New['VAR_0342'].replace(to_replace='-1', value='01')
    Actual_DS_New['VAR_0342_1'] = Actual_DS_New['VAR_0342'].str[:1]
    Actual_DS_New['VAR_0342_2'] = Actual_DS_New['VAR_0342'].str[1:2]

    #-------------------------------------------------------------------------------------------------------------------
    #All remaining must be label encoded
    label_enc_cols = ['VAR_0237',  'VAR_0342','VAR_0342_1','VAR_0342_2',  'VAR_0466']
    for i in range(len(label_enc_cols)):
            lbl = preprocessing.LabelEncoder()
            lbl.fit((list(Train_DS_New[label_enc_cols[i]].astype(str)) + list(Actual_DS_New[label_enc_cols[i]].astype(str))))
            Train_DS_New[label_enc_cols[i]] = lbl.transform(Train_DS_New[label_enc_cols[i]].astype(str))
            Actual_DS_New[label_enc_cols[i]] = lbl.transform(Actual_DS_New[label_enc_cols[i]].astype(str))

    print("Ending Categorical conversion....")

    print(np.shape(Train_DS_New))
    print(np.shape(Actual_DS_New))

    ####################################################################################################################
    print("Starting Misc Variable conversion....")

    Train_DS_New[cols_others]  = Train_DS[cols_others].fillna('00')
    Actual_DS_New[cols_others] = Actual_DS[cols_others].fillna('00')

    #-------------------------------------------------------------------------------------------------------------------
    #looks like VAR_0200 is city description, this will anyway be captured in state code and Zip codes. So delete it
    Train_DS_New = Train_DS_New.drop(['VAR_0200'], axis = 1)
    Actual_DS_New = Actual_DS_New.drop(['VAR_0200'], axis = 1)
    #-------------------------------------------------------------------------------------------------------------------
    #VAR_0404 is the profession / designation . May be this requires more cleaning
    #*******************************************************************************************************************

    #-------------------------------------------------------------------------------------------------------------------
    #VAR_0467  , Diff types of Discharges , make everything same
    Train_DS_New['VAR_0467'] = Train_DS_New['VAR_0467'].replace(to_replace='Discharge NA', value='Discharged')
    Actual_DS_New['VAR_0467'] = Actual_DS_New['VAR_0467'].replace(to_replace='Discharge NA', value='Discharged')

    #-------------------------------------------------------------------------------------------------------------------
    #VAR_0493 is the profession / designation . May be this requires more cleaning
    #*******************************************************************************************************************

    #-------------------------------------------------------------------------------------------------------------------
    # VAR_1934 - only 5 values - Devectorize / one hot encoding
    for column in ['VAR_1934']:
        dummies = pd.get_dummies(Train_DS_New[column])
        Train_DS_New[dummies.columns] = dummies

        dummies = pd.get_dummies(Actual_DS_New[column])
        Actual_DS_New[dummies.columns] = dummies

    Train_DS_New = Train_DS_New.drop(['VAR_1934'], axis = 1)
    Actual_DS_New = Actual_DS_New.drop(['VAR_1934'], axis = 1)

    #-------------------------------------------------------------------------------------------------------------------
    #All remaining must be label encoded
    label_enc_cols = ['VAR_0214','VAR_0404','VAR_0467','VAR_0493']
    for i in range(len(label_enc_cols)):
            lbl = preprocessing.LabelEncoder()
            lbl.fit((list(Train_DS_New[label_enc_cols[i]].astype(str)) + list(Actual_DS_New[label_enc_cols[i]].astype(str))))
            Train_DS_New[label_enc_cols[i]] = lbl.transform(Train_DS_New[label_enc_cols[i]].astype(str))
            Actual_DS_New[label_enc_cols[i]] = lbl.transform(Actual_DS_New[label_enc_cols[i]].astype(str))

    print("Ending Misc Variable conversion....")

    ####################################################################################################################

    print(np.shape(Train_DS_New))
    print(np.shape(Actual_DS_New))

    ####################################################################################################################
    #Any Additional Data cleansing before label encoding

    # Get the attribute frequency count , then sum it up and add it as a new column
    # New_DS = pd.concat([Train_DS, Actual_DS])
    #
    # Train_DS_T = Train_DS
    # Actual_DS_T = Actual_DS
    #
    # for i in range(Train_DS.shape[1]):
    #     print(i)
    #     cols = columns[i]
    #     Feature_count = (New_DS[cols].value_counts()/  New_DS.shape[0]).reset_index()
    #     Feature_count.columns = [cols, cols+'_new']
    #     Train_DS = Train_DS.merge(Feature_count, on=cols)
    #     Actual_DS = Actual_DS.merge(Feature_count, on=cols)
    #
    # Train_DS = Train_DS.drop(columns, axis = 1)
    # Actual_DS = Actual_DS.drop(columns, axis = 1)
    ####################################################################################################################
    #Train_DS['sum'] = Train_DS_T.sum(axis=1)
    #Actual_DS['sum'] = Actual_DS_T.sum(axis=1)

        # print(Feature_count)
        # print(Train_DS[cols].head())
        # print(Train_DS[cols+'_new'].head())
        #print(Train_DS['VAR_0001'])

    ####################################################################################################################
    #Feature selection

    # selector = GenericUnivariateSelect(score_func=f_classif, mode = 'percentile', param=90)
    # selector.fit(Train_DS, y)
    # Train_DS = selector.transform(Train_DS)
    # Actual_DS = selector.transform(Actual_DS)
    #
    # print(np.shape(Train_DS))
    # print(np.shape(Actual_DS))

    #pd.DataFrame(Actual_DS).to_csv(file_path+'Actual_DS_Temp2.csv')


    #Feature Selection using wrapper method (l1 regularization)
    # clf = LinearSVC(C=0.001, penalty="l1", dual=False)
    # #clf = LogisticRegression(C=0.001, penalty="l1", dual=False)
    # clf.fit(Train_DS, y)
    # Train_DS_New  = clf.transform(Train_DS)
    # Actual_DS_New = clf.transform(Actual_DS)

    # imp = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)
    # Train_DS_New  = imp.fit_transform(Train_DS_New)
    # Actual_DS_New = imp.fit_transform(Actual_DS_New)

    #Delete features with 0 importance value using random forest check
    # Train_DS1 = Train_DS_New
    # Irrelevant_feat = list(Featimp_DS[Featimp_DS['imp'] <= 0 ]['col'])
    # Irrelevant_feat = ['VAR_0195','VAR_0194','VAR_0193','VAR_0192','VAR_0214','VAR_0191','VAR_0181','VAR_0098','VAR_0139','VAR_0130','VAR_1012','VAR_0114']
    # Train_DS_New = Train_DS_New.drop(Irrelevant_feat, axis = 1)
    # Actual_DS_New = Actual_DS_New.drop(Irrelevant_feat, axis = 1)

    ####################################################################################################################
    #Shuffle the Dataset
    Train_DS_New, y = shuffle(Train_DS_New, y, random_state=21)

    # #Setting Standard scaler for data
    stdScaler = StandardScaler(with_mean=True, with_std=True)
    stdScaler.fit(Train_DS_New,y)
    Train_DS_New = stdScaler.transform(Train_DS_New)
    Actual_DS_New = stdScaler.transform(Actual_DS_New)

    #apply PCA
    # print("PCA = 2030")
    # #pca = PCA(n_components=2030, whiten=True)
    # pca = TruncatedSVD(n_components=2000,algorithm='arpack')
    # pca.fit(Train_DS,y)
    # Train_DS = pca.transform(Train_DS)
    # Actual_DS = pca.transform(Actual_DS)
    # print(pca.components_)


    # print("LDA = 2030")
    # lda = LDA(n_components=100)
    # lda.fit(Train_DS,y)
    # Train_DS = lda.transform(Train_DS)
    # Actual_DS = lda.transform(Actual_DS)

    print(np.shape(Train_DS_New))
    print(np.shape(Actual_DS_New))

    print("***************Ending Data cleansing***************")

    return Train_DS_New, Actual_DS_New, y

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

        #Best on grid ::::   CV:
        # clf = xgb.XGBClassifier(n_estimators=500,max_depth=4,learning_rate=0.1,nthread=2,min_child_weight=11,
        #                      subsample=0.8,colsample_bytree=0.7,silent=True, gamma = 0.6)

        #from Kaggle
        clf = xgb.XGBClassifier(n_estimators=500,max_depth=9,learning_rate=0.01,nthread=2,min_child_weight=6,
                             subsample=0.7,colsample_bytree=0.5,silent=True, gamma = 4)

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)


        # clf = xgb.XGBClassifier(n_estimators=2000,max_depth=4,learning_rate=0.1,nthread=2,min_child_weight=11,
        #                      subsample=0.8,colsample_bytree=0.7,silent=True, gamma = 0.6)

        #from Kaggle (https://www.kaggle.com/c/springleaf-marketing-response/forums/t/16808/time-window-variables-features)
        #clf = xgb.XGBClassifier(n_estimators=2000,max_depth=10,learning_rate=0.005,nthread=2,min_child_weight=11,
        #                     subsample=0.8,colsample_bytree=0.4,silent=True, gamma = 0.6)

        #from Kaggle
        clf = xgb.XGBClassifier(n_estimators=2000,max_depth=9,learning_rate=0.01,nthread=2,min_child_weight=6,
                             subsample=0.7,colsample_bytree=0.5,silent=True, gamma = 4)

        clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

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

        file_path = 'C:/Python/Others/data/Kaggle/Springleaf_Marketing_Response/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Springleaf_Marketing_Response/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    #Train_DS      = pd.read_csv(file_path+'train.csv',sep=',')
    #Actual_DS     = pd.read_csv(file_path+'test.csv',sep=',')

    Train_DS      = pd.read_csv(file_path+'train_25000.csv',sep=',', index_col=0,nrows = 5000 ).reset_index(drop=True)
    Actual_DS     = pd.read_csv(file_path+'test_25000.csv',sep=',', index_col=0,nrows = 5000).reset_index(drop=True)

    Sample_DS     = pd.read_csv(file_path+'sample_submission.csv',sep=',')
    Filter_DS     = pd.read_csv(file_path+'Min_Max_DS_Analysis2.csv',sep=',')
    Featimp_DS    = pd.read_csv(file_path+'feature_imp.csv',sep=',')

    Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS, Filter_DS)

    #pred_Actual = XGB_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    pred_Actual  = RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    #pred_Actual  = Misc_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)