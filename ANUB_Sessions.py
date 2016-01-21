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
def Data_Munging(Session_DS):

    print("***************Starting Data cleansing***************")

    ##----------------------------------------------------------------------------------------------------------------##
    #Delete all rows with NULL user_id's
    #Session_DS = Session_DS.head(100000)
    Session_DS = Session_DS[Session_DS['user_id'].notnull()]

    ##----------------------------------------------------------------------------------------------------------------##
    #Delete unnecessary fields
    Session_DS = Session_DS.drop(['device_type','secs_elapsed'], axis = 1)

    Session_DS['action_type']   = Session_DS['action_type'].replace([np.inf, -np.inf,np.nan], 'NULL')
    Session_DS['action']        = Session_DS['action'].replace([np.inf, -np.inf,np.nan], 'NULL')
    Session_DS['action_detail'] = Session_DS['action_detail'].replace([np.inf, -np.inf,np.nan], 'NULL')

    New_Session_DS = pd.DataFrame()
    New_Session_DS['id'] = Session_DS.user_id.unique()

    ##----------------------------------------------------------------------------------------------------------------##
    # Create dummy's for action type

    dummy_type = 'action_type'
    unique_type = list(Session_DS[dummy_type].unique())

    for i in range(len(unique_type)):
        print("%d -of- %d - iteration - %s" %(i+1,len(unique_type),str(unique_type[i])))
        dummy_value = str(unique_type[i])
        temp = Session_DS[Session_DS[dummy_type]==dummy_value].groupby('user_id').action_type.count().reset_index()
        temp.columns = ['id',dummy_type+'_'+dummy_value]
        New_Session_DS = New_Session_DS.merge(temp,on=['id'], how='left')

    New_Session_DS = New_Session_DS.set_index('id')
    print(New_Session_DS.head())
    pd.DataFrame(New_Session_DS).to_csv(file_path+'New_Session_DS_'+dummy_type+'.csv')

    print("***************Ending Data cleansing***************")

    return Session_DS

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
    Session_DS    = pd.read_csv(file_path+'sessions.csv',sep=',')

    ##----------------------------------------------------------------------------------------------------------------##
    Session_DS =  Data_Munging(Session_DS)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys)