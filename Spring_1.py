import requests
import numpy as np
import scipy as sp
import sys
import platform
import pandas as pd
from time import time
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.ensemble import RandomForestClassifier ,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import re
import warnings
from math import sqrt, exp, log
from csv import DictReader
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV, ParameterSampler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint as sp_randint
from sklearn import decomposition, pipeline, metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score,roc_curve,auc
#import xgboost as xgb
import collections
import ast
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LogisticRegression, \
    Perceptron,RidgeCV, TheilSenRegressor
from datetime import date,timedelta,datetime as dt
import datetime
from sklearn.feature_selection import SelectKBest,SelectPercentile, f_classif, GenericUnivariateSelect
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.lda import LDA
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
########################################################################################################################
#Springleaf Marketing Response
########################################################################################################################
#--------------------------------------------Algorithm : Random Forest :------------------------------------------------
#Random Forest :
#--------------------------------------------Algorithm : XGB------------------------------------------------------------
#XGB :

#--------------------------------------------Suggestions, Ideas---------------------------------------------------------
#Suggestions, Ideas

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

    #ss = StratifiedShuffleSplit(y, n_iter=5,test_size=0.2, random_state=42, indices=None)
    ss = KFold(len(y), n_folds=5,shuffle=False,indices=None,random_state=42)
    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        #print(X_test[5,:])

        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')
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

    y = Train_DS.target.values

    cols = list(Filter_DS['feature'])
    Train_DS = Train_DS[cols]
    print(Train_DS)

    sys.exit(0)
    Train_DS = Train_DS.drop(['target','ID','VAR_0044'], axis = 1)
    Actual_DS = Actual_DS.drop(['ID','VAR_0044'], axis = 1)


    ####################################################################################################################
    #Delete columns as per duplicates.....
    Train_DS = Train_DS.drop(['VAR_0200','VAR_0154','VAR_0164','VAR_0155','VAR_0165'], axis = 1)
    Actual_DS = Actual_DS.drop(['VAR_0200','VAR_0154','VAR_0164','VAR_0155','VAR_0165'], axis = 1)

    ####################################################################################################################
    #Find Min max of each column

    # New_DS = pd.concat([Train_DS, Actual_DS])
    #
    # df_uniq = pd.DataFrame({func.__name__:New_DS.apply(func) for func in (pd.Series.nunique, pd.Series.count)}).reset_index()
    # df_uniq.columns = ['VAR','Count', 'Uniq',]
    # df_uniq = df_uniq.drop(['Count'], axis = 1)
    #
    # df_min = New_DS.min().reset_index()
    # df_min.columns = ['VAR', 'Min']
    # df_max = New_DS.max().reset_index()
    # df_max.columns = ['VAR', 'Max']
    #
    # New_DF =    df_min.merge(df_max, on='VAR')
    # New_DF =    New_DF.merge(df_uniq, on='VAR')
    # New_DF.to_csv(file_path+'Min_Max_DS.csv')
    # print(New_DF.head())
    # sys.exit(0)
    ####################################################################################################################
    #sometimes boolean is coming as Object (string). So convert it
    #Train_DS = Train_DS.replace(to_replace=False, value='0')
    #Train_DS = Train_DS.replace(to_replace=True, value='1')

    ####################################################################################################################
    #For the date fields in Train and Actual
    datecolumns = ['VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159', 'VAR_0166', 'VAR_0167', 'VAR_0168','VAR_0169', 'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179', 'VAR_0204', 'VAR_0217']
    monthDict={'JAN':'01', 'FEB':'02', 'MAR':'03', 'APR':'04', 'MAY':'05', 'JUN':'06', 'JUL':'07', 'AUG':'08', 'SEP':'09', 'OCT':'10', 'NOV':'11', 'DEC':'12'}

    #clean up all date fields, extract DD,MMM,YYYY and combine to a single date for further use
    for datecol in datecolumns:

        Train_DS[datecol+'_DD'] = Train_DS[datecol].str[:2].fillna(1).astype(int)
        Train_DS[datecol+'_MM'] = Train_DS[datecol].str[2:5].replace(monthDict).fillna(1).astype(int)
        Train_DS[datecol+'_YY'] = Train_DS[datecol].str[5:7].fillna(1).astype(int).apply(lambda x: x+2000)
        #Train_DS[datecol+'_YY'] = Train_DS[datecol].str[5:7].fillna(1).astype(int)
        Train_DS[datecol]       = pd.to_datetime(Train_DS[datecol+'_YY']*10000+Train_DS[datecol+'_MM']*100+Train_DS[datecol+'_DD'],format='%Y%m%d')
        #temp = pd.DatetimeIndex(Train_DS[datecol])
        #Train_DS[datecol+'_DW'] = temp.dayofweek
        #Train_DS[datecol+'_WK'] = temp.week

        Actual_DS[datecol+'_DD'] = Actual_DS[datecol].str[:2].fillna(1).astype(int)
        Actual_DS[datecol+'_MM'] = Actual_DS[datecol].str[2:5].replace(monthDict).fillna(1).astype(int)
        Actual_DS[datecol+'_YY'] = Actual_DS[datecol].str[5:7].fillna(1).astype(int).apply(lambda x: x+2000)
        #Actual_DS[datecol+'_YY'] = Actual_DS[datecol].str[5:7].fillna(1).astype(int)
        Actual_DS[datecol]       = pd.to_datetime(Actual_DS[datecol+'_YY']*10000+Actual_DS[datecol+'_MM']*100+Actual_DS[datecol+'_DD'],format='%Y%m%d')
        #temp = pd.DatetimeIndex(Actual_DS[datecol])
        #Actual_DS[datecol+'_DW'] = temp.dayofweek
        #Actual_DS[datecol+'_WK'] = temp.week

        #Get Today - date --day values
        Train_DS[datecol+'_today'] = (datetime.datetime.today() - Train_DS[datecol]).astype('timedelta64[D]')
        Actual_DS[datecol+'_today'] = (datetime.datetime.today() - Actual_DS[datecol]).astype('timedelta64[D]')

    # Get Date Differences
    for index , datecol in enumerate(datecolumns):
        j = index + 1
        for i in range(j , len(datecolumns)):
            newcol = datecolumns[index]+"_"+datecolumns[i]
            Train_DS[newcol] = (Train_DS[datecolumns[index]] - Train_DS[datecolumns[i]]).astype('timedelta64[D]')
            Actual_DS[newcol] = (Actual_DS[datecolumns[index]] - Actual_DS[datecolumns[i]]).astype('timedelta64[D]')

    Train_DS = Train_DS.drop(datecolumns, axis = 1)
    Actual_DS = Actual_DS.drop(datecolumns, axis = 1)
    print("Date cleaned up and converted")

    ####################################################################################################################
    #Delete columns with only one Unique values

    columns = Train_DS.columns
    col_types = (Train_DS.dtypes).reset_index(drop=True)

    #Get column unique count
    unique_cols = []
    for j in range(Train_DS.shape[1]):
        if (len(Train_DS[columns[j]].value_counts(dropna=True))) <= 1:
            unique_cols.append(columns[j])

    Train_DS = Train_DS.drop(unique_cols, axis = 1)
    Actual_DS = Actual_DS.drop(unique_cols, axis = 1)

    print("Unique columns deleted")

    ####################################################################################################################
    #Assign Default values for each  data type column

    columns = Train_DS.columns
    col_types = (Train_DS.dtypes).reset_index(drop=True)

    for i in range(Train_DS.shape[1]):
        if col_types[i] =='object':
            Actual_DS[str(columns[i])] = Actual_DS[str(columns[i])].fillna('0')
            Train_DS[str(columns[i])] = Train_DS[str(columns[i])].fillna('0')
        if col_types[i] =='bool':
            Actual_DS[str(columns[i])] = Actual_DS[str(columns[i])].fillna(False)
            Train_DS[str(columns[i])] = Train_DS[str(columns[i])].fillna(False)
        else:
            Actual_DS[str(columns[i])] = Actual_DS[str(columns[i])].fillna(0)
            Train_DS[str(columns[i])] = Train_DS[str(columns[i])].fillna(0)

    print("Assigned Default values for each  data type column")

    ####################################################################################################################

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


    # for i in range(Train_DS.shape[1]):
    #     print(Train_DS[str(columns[i])].min())

    # Train_DS['VAR_0212_ZIP'] = Train_DS['VAR_0212'].apply(lambda x: int(x/1000000)).astype(int)
    # Train_DS['VAR_0212_ZIC'] = Train_DS['VAR_0212'].apply(lambda x: int((x%1000000)/100)).astype(int)
    # Train_DS['VAR_0212_REM'] = Train_DS['VAR_0212'].apply(lambda x: int(x%100)).astype(int)
    #
    # Actual_DS['VAR_0212_ZIP'] = Actual_DS['VAR_0212'].apply(lambda x: int(x/1000000)).astype(int)
    # Actual_DS['VAR_0212_ZIC'] = Actual_DS['VAR_0212'].apply(lambda x: int((x%1000000)/100)).astype(int)
    # Actual_DS['VAR_0212_REM'] = Actual_DS['VAR_0212'].apply(lambda x: int(x%100)).astype(int)

    #Train_DS = Train_DS.replace(to_replace=-1, value=0)
    #Actual_DS = Actual_DS.replace(to_replace=-1, value=0)
    ####################################################################################################################
    #label encode the categorical variables
    columns = Train_DS.columns
    col_types = (Train_DS.dtypes).reset_index(drop=True)

    Train_DS1 = Train_DS.head()
    Train_DS = np.array(Train_DS)
    Actual_DS = np.array(Actual_DS)

    #print(set(Train_DS[:,7]))
    #print((list(set(list(Train_DS[:,7]) ))))

    print("Starting label encoding")
    for i in range(Train_DS.shape[1]):
        if col_types[i] =='object'  or col_types[i] =='bool':
            lbl = preprocessing.LabelEncoder()
            lbl.fit((list(Train_DS[:,i].astype(str)) + list(Actual_DS[:,i].astype(str))))
            Train_DS[:,i] = lbl.transform(Train_DS[:,i].astype(str))
            Actual_DS[:,i] = lbl.transform(Actual_DS[:,i].astype(str))
    print("Ending label encoding")

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))
    ####################################################################################################################
    #Shuffle the Dataset

    #Train_DS, y = shuffle(Train_DS, y, random_state=42)

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

    # #Setting Standard scaler for data
    stdScaler = StandardScaler(with_mean=True, with_std=True)
    stdScaler.fit(Train_DS,y)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

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

    ####################################################################################################################
    #Try Clustering

    #clust = AgglomerativeClustering(n_clusters=10,linkage='ward')
    clust = FeatureAgglomeration(n_clusters=1000,linkage='ward')

    clust.fit(Train_DS)
    #clust.fit(Train_DS,y)
    Train_DS = clust.transform(Train_DS)
    Actual_DS = clust.transform(Actual_DS)

    agglo = pd.DataFrame()
    agglo['cluster'] = clust.labels_
    agglo['feature'] = columns
    agglo = agglo.sort(['cluster'], ascending=True).reset_index(drop=True)
    print(agglo)
    sys.exit(0)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

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

        clf = RandomForestClassifier(n_jobs=-1, n_estimators=100)

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        sys.exit(0)
        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        clf.fit(Train_DS, y)

        #
        # feature = pd.DataFrame()
        # feature['imp'] = clf.feature_importances_
        # feature['col'] = Train_DS1.columns
        # feature = feature.sort(['imp'], ascending=False).reset_index(drop=True)
        # print(feature)

        sys.exit(0)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_RFC.csv', index_label='ID')

    print("***************Ending Random Forest Classifier***************")
    return pred_Actual


########################################################################################################################
#XGB Regressor
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

        #starting model
        clf = xgb.XGBClassifier(n_estimators=100,max_depth=15,learning_rate=0.1,nthread=2,min_child_weight=1,
                             subsample=0.6,colsample_bytree=0.7,silent=True, gamma = 2)

        #starting model , got i from kaggle script
        clf = xgb.XGBClassifier(n_estimators=100,max_depth=9,learning_rate=0.01,nthread=2,min_child_weight=6,
                             subsample=0.7,colsample_bytree=0.5,silent=True, gamma = 4)


        #CV:0.7719 (N_est=100)
        clf = xgb.XGBClassifier(n_estimators=1000,nthread=-1)


        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)


        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set

    preds = pd.DataFrame(pred_Actual, index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_xgb.csv', index_label='ID')

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

        clf = LogisticRegression()

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        sys.exit(0)
        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        clf.fit(Train_DS, y)

        #
        # feature = pd.DataFrame()
        # feature['imp'] = clf.feature_importances_
        # feature['col'] = Train_DS1.columns
        # feature = feature.sort(['imp'], ascending=False).reset_index(drop=True)
        # print(feature)

        sys.exit(0)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_RFC.csv', index_label='ID')

    print("***************Ending Random Forest Classifier***************")
    return pred_Actual


########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path

    if(platform.system() == "Windows"):

        file_path = 'C:/Python/Others/data/Kaggle/Springleaf_Marketing_Response/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Springleaf_Marketing_Response/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    #Train_DS      = pd.read_csv(file_path+'train.csv',sep=',')
    #Actual_DS     = pd.read_csv(file_path+'test.csv',sep=',')

    Train_DS      = pd.read_csv(file_path+'train_10000.csv',sep=',', index_col=0,nrows=5000)
    Actual_DS     = pd.read_csv(file_path+'test_10000.csv',sep=',', index_col=0,nrows=7500)

    Sample_DS     = pd.read_csv(file_path+'sample_submission.csv',sep=',')
    Filter_DS     = pd.read_csv(file_path+'Min_Max_DS_Analysis3.csv',sep=',')

    Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS, Filter_DS)

    #pred_Actual = XGB_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    pred_Actual  = RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    #pred_Actual  = Misc_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)