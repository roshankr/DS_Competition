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
import time as tm
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
from sklearn.metrics.pairwise import *
from sklearn import preprocessing

########################################################################################################################
#Walmart Recruiting: Trip Type Classification - Utility program
# The out of this can be directly merged with Train and Actual DS
########################################################################################################################

########################################################################################################################
def get_FinelineNumber(row, i):
    value = 0
    if row['FinelineNumber'] == i:
        value= 1
    return i

########################################################################################################################
#Get_High_Lowest_Contributors
########################################################################################################################
def Get_High_Lowest_Contributors(New_DS):

    print("Get_High_Lowest_Contributors")
    # Find highest and lowest contributor in sales for each visit , dept wise
    Temp = New_DS[New_DS['ScanCount'] > 0 ].groupby(['VisitNumber','DepartmentDescription']).sum().ScanCount.reset_index()
    DD_Max_per_visit_buy = (Temp.sort(columns=['VisitNumber','ScanCount'],ascending=[True,False])).groupby(['VisitNumber']).first().reset_index()
    DD_Max_per_visit_buy = DD_Max_per_visit_buy.drop(['ScanCount'], axis = 1)
    DD_Max_per_visit_buy.columns = ['VisitNumber','DD_Max_per_visit_buy']

    Temp = New_DS[New_DS['ScanCount'] < 0 ].groupby(['VisitNumber','DepartmentDescription']).sum().ScanCount.reset_index()
    Temp['ScanCount'] = Temp['ScanCount']*-1
    DD_Max_per_visit_ret = (Temp.sort(columns=['VisitNumber','ScanCount'],ascending=[True,False])).groupby(['VisitNumber']).first().reset_index()
    DD_Max_per_visit_ret = DD_Max_per_visit_ret.drop(['ScanCount'], axis = 1)
    DD_Max_per_visit_ret.columns = ['VisitNumber','DD_Max_per_visit_ret']

    # Find highest and lowest contributor in sales for each visit , Fineline number wise
    Temp = New_DS[New_DS['ScanCount'] > 0 ].groupby(['VisitNumber','FinelineNumber']).sum().ScanCount.reset_index()
    FN_Max_per_visit_buy = (Temp.sort(columns=['VisitNumber','ScanCount'],ascending=[True,False])).groupby(['VisitNumber']).first().reset_index()
    FN_Max_per_visit_buy = FN_Max_per_visit_buy.drop(['ScanCount'], axis = 1)
    FN_Max_per_visit_buy.columns = ['VisitNumber','FN_Max_per_visit_buy']

    Temp = New_DS[New_DS['ScanCount'] < 0 ].groupby(['VisitNumber','FinelineNumber']).sum().ScanCount.reset_index()
    Temp['ScanCount'] = Temp['ScanCount']*-1
    FN_Max_per_visit_ret = (Temp.sort(columns=['VisitNumber','ScanCount'],ascending=[True,False])).groupby(['VisitNumber']).first().reset_index()
    FN_Max_per_visit_ret = FN_Max_per_visit_ret.drop(['ScanCount'], axis = 1)
    FN_Max_per_visit_ret.columns = ['VisitNumber','FN_Max_per_visit_ret']

    High_Lowest_Contributors = pd.DataFrame(New_DS['VisitNumber'].unique())
    High_Lowest_Contributors.columns=['VisitNumber']

    High_Lowest_Contributors = High_Lowest_Contributors.merge(DD_Max_per_visit_buy, on ='VisitNumber', how='left')
    High_Lowest_Contributors = High_Lowest_Contributors.merge(DD_Max_per_visit_ret, on ='VisitNumber', how='left')
    High_Lowest_Contributors = High_Lowest_Contributors.merge(FN_Max_per_visit_buy, on ='VisitNumber', how='left')
    High_Lowest_Contributors = High_Lowest_Contributors.merge(FN_Max_per_visit_ret, on ='VisitNumber', how='left')
    High_Lowest_Contributors = High_Lowest_Contributors.fillna(0)

    print(np.shape(High_Lowest_Contributors))

    del New_DS

    return High_Lowest_Contributors

########################################################################################################################
#Get_Upc dummy
########################################################################################################################
def Get_Upc_dummy(New_DS):

    print("Get_Upc_dummy")
    Test_DS = (New_DS[New_DS['Upc']!=0].groupby('Upc').VisitNumber.count()).reset_index()
    Test_DS.columns=['Upc','count']
    Test_DS = Test_DS.sort(columns='count',ascending=False).reset_index(drop=True)
    Test_DS = Test_DS.head(1000)

    dummies = pd.get_dummies(Test_DS['Upc'])
    Upc_cols = [ 'Upc_1000_'+str(s) for s in list(dummies.columns)]
    Test_DS[Upc_cols] = dummies
    cols = ['VisitNumber','Upc']
    Test_DS = Test_DS.merge(New_DS[cols],on='Upc')

    Test_DS = Test_DS.sort(columns='VisitNumber')
    Test_DS = Test_DS.drop(['count','Upc'], axis = 1)
    Test_DS = Test_DS.groupby('VisitNumber').sum().reset_index()

    print(Test_DS['VisitNumber'].nunique())
    print(np.shape(Test_DS))

    return Test_DS

########################################################################################################################
#Get_FinelineNumber dummy
########################################################################################################################
def Get_Fineline_dummy(New_DS):

    print("Get_FinelineNumber_dummy")
    Test_DS = (New_DS[New_DS['FinelineNumber']!=0].groupby('FinelineNumber').VisitNumber.count()).reset_index()
    Test_DS.columns=['FinelineNumber','count']
    Test_DS = Test_DS.sort(columns='count',ascending=False).reset_index(drop=True)
    Test_DS = Test_DS.head(1000)

    dummies = pd.get_dummies(Test_DS['FinelineNumber'])
    FinelineNumber_cols = [ 'FinelineNumber_1000_'+str(s) for s in list(dummies.columns)]
    Test_DS[FinelineNumber_cols] = dummies
    cols = ['VisitNumber','FinelineNumber']
    Test_DS = Test_DS.merge(New_DS[cols],on='FinelineNumber')
    Test_DS = Test_DS.sort(columns='VisitNumber')
    Test_DS = Test_DS.drop(['count','FinelineNumber'], axis = 1)
    Test_DS = Test_DS.groupby('VisitNumber').sum().reset_index()

    print(Test_DS['VisitNumber'].nunique())
    print(np.shape(Test_DS))

    return Test_DS

########################################################################################################################
#Get_Similarity Matrix
########################################################################################################################
def Get_similarity_matrix(Train_DS,y):

    print("Get DD Similarity Matrix")

    Train_DS['TripType']= y
    Test_DS = Train_DS.groupby(['TripType','DepartmentDescription']).sum().ScanCount.reset_index()
    Test_DS.columns=['TripType','DepartmentDescription','scan_sum']

    dummies = pd.get_dummies(list(Test_DS['TripType']))
    VN_cols = [ 'VN_'+str(s) for s in list(dummies.columns)]
    Test_DS[VN_cols] = dummies

    for i in range(len(VN_cols)):
        Test_DS[VN_cols[i]] = (Test_DS[VN_cols[i]] *Test_DS['scan_sum'])

    Test_DS = Test_DS.drop(['TripType','scan_sum'], axis = 1)
    Test_DS = Test_DS.groupby('DepartmentDescription').sum().reset_index()
    Test_DS = Test_DS.sort(columns='DepartmentDescription',ascending=True)

    Test_New = Test_DS[VN_cols]

    cos_dist_T  = pd.DataFrame(cosine_similarity(Test_New))
    cos_dist_T = 1 / cos_dist_T

    cos_dist_T = cos_dist_T.replace([np.inf, -np.inf], 1)

    cos_dist_sum = cos_dist_T.sum(axis=0)

    cos_dist_T = (cos_dist_T / cos_dist_sum)

    print(np.shape(cos_dist_T))

    return cos_dist_T

########################################################################################################################
#Get DD One hot encodin
########################################################################################################################
def DD_onehot_encoding(New_DS, Train_DS, y):

    cos_dist_T = Get_similarity_matrix(Train_DS,y)

    #one hot encoding for DepartmentDescription
    print("one hot encoding - DepartmentDescription at Time: %s" %(tm.strftime("%H:%M:%S")))

    dummies = pd.get_dummies(New_DS['DepartmentDescription'])
    DeptDesc_cols = [ 'DD'+"_buy1_"+str(s) for s in list(dummies.columns)]

    sim_dd_buy = cos_dist_T
    sim_dd_buy.columns = DeptDesc_cols
    sim_dd_buy = sim_dd_buy.reset_index()

    cols = ['VisitNumber','ScanCount','DepartmentDescription']
    New_DS = New_DS[cols].merge(sim_dd_buy,left_on='DepartmentDescription',right_on='index',how='left')
    New_DS = New_DS.drop(['index'], axis = 1)

    #get "buying" qty for  DepartmentDescription
    Temp_Scan = pd.DataFrame()
    Temp_Scan['ScanCount'] = New_DS ['ScanCount']
    Temp_Scan['ScanCount'] = np.where(New_DS ['ScanCount']>= 0,New_DS ['ScanCount'],0).astype(int)

    for i in range(len(DeptDesc_cols)):
        New_DS[DeptDesc_cols[i]] = New_DS[DeptDesc_cols[i]] * Temp_Scan ['ScanCount']

    del sim_dd_buy
    ##----------------------------------------------------------------------------------------------------------------##

    #one hot encoding for DepartmentDescription - Return
    dummies = pd.get_dummies(New_DS['DepartmentDescription'])
    DeptDesc_cols = [ 'DD'+"_ret1_"+str(s) for s in list(dummies.columns)]

    sim_dd_ret = cos_dist_T
    sim_dd_ret.columns = DeptDesc_cols
    sim_dd_ret = sim_dd_ret.reset_index()

    New_DS = New_DS.merge(sim_dd_ret,left_on='DepartmentDescription',right_on='index',how='left')
    New_DS = New_DS.drop(['index'], axis = 1)

    #get "return" qty for  DepartmentDescription
    Temp_Scan['ScanCount'] = New_DS ['ScanCount']
    Temp_Scan['ScanCount'] = np.where(New_DS ['ScanCount'] < 0,New_DS ['ScanCount']*-1,0).astype(int)

    for i in range(len(DeptDesc_cols)):
        New_DS[DeptDesc_cols[i]] = New_DS[DeptDesc_cols[i]] * Temp_Scan ['ScanCount']

    del sim_dd_ret
    ##----------------------------------------------------------------------------------------------------------------##
    New_DS = New_DS.drop(['ScanCount','DepartmentDescription'], axis = 1)
    New_DS = New_DS.groupby('VisitNumber').sum().reset_index()

    print(np.shape(New_DS))

    #pd.DataFrame(New_DS).to_csv(file_path+'New_DS.csv')

    return New_DS

########################################################################################################################
#Get_Best_Trip_for_DD
########################################################################################################################
def Get_Best_Trip_for_DD(Train_DS,y):

    print("Get_Best Trip Type")
    Train_DS['TripType']= y

    Temp = Train_DS[Train_DS['ScanCount'] > 0 ].groupby(['TripType','DepartmentDescription']).sum().ScanCount.reset_index()
    Temp = Temp.sort(['DepartmentDescription','ScanCount'],ascending=False)
    Temp.columns=['TripType','DepartmentDescription','ScanCount_indi']

    Temp1 = Train_DS[Train_DS['ScanCount'] > 0 ].groupby(['DepartmentDescription']).sum().ScanCount.reset_index()
    Temp1.columns=['DepartmentDescription','ScanCount_tot']

    Temp = Temp.merge(Temp1,on='DepartmentDescription')
    Temp['ScanCount_avg'] = Temp['ScanCount_indi'] / Temp['ScanCount_tot']
    Temp = Temp.drop(['ScanCount_indi','ScanCount_tot'], axis = 1)

    Temp = Temp.sort(columns=['DepartmentDescription','ScanCount_avg'],ascending=[True,False]).groupby(['DepartmentDescription']).first().reset_index()
    Temp.columns=['DD_Max_per_visit_buy','TripType_Best','ScanCount_Best']
    Temp = Temp.fillna(0)

    return Temp

########################################################################################################################
#Get_Best_Trip_for_DD
########################################################################################################################
def Get_Item_quantity(New_DS):

    print("Get Item Quantity")
    # Find highest and lowest contributor in sales for each visit , dept wise
    Item_quantity = New_DS[New_DS['ScanCount'] == 1 ].groupby(['VisitNumber']).count().Upc.reset_index()
    Item_quantity.columns = ['VisitNumber','count_buy_1']

    Temp = New_DS[New_DS['ScanCount'] == 2 ].groupby(['VisitNumber']).count().Upc.reset_index()
    Temp.columns = ['VisitNumber','count_buy_2']
    Item_quantity = Item_quantity.merge(Temp,on='VisitNumber',how='left')

    Temp = New_DS[(New_DS['ScanCount'] > 2) & (New_DS['ScanCount'] <= 5) ].groupby(['VisitNumber']).count().Upc.reset_index()
    Temp.columns = ['VisitNumber','count_buy_2_5']
    Item_quantity = Item_quantity.merge(Temp,on='VisitNumber',how='left')

    Temp = New_DS[(New_DS['ScanCount'] > 5) & (New_DS['ScanCount'] <= 10) ].groupby(['VisitNumber']).count().Upc.reset_index()
    Temp.columns = ['VisitNumber','count_buy_5_10']
    Item_quantity = Item_quantity.merge(Temp,on='VisitNumber',how='left')

    Temp = New_DS[(New_DS['ScanCount'] > 10) ].groupby(['VisitNumber']).count().Upc.reset_index()
    Temp.columns = ['VisitNumber','count_buy_10']
    Item_quantity = Item_quantity.merge(Temp,on='VisitNumber',how='left')

    Temp = New_DS[New_DS['ScanCount'] == -1 ].groupby(['VisitNumber']).count().Upc.reset_index()
    Temp.columns = ['VisitNumber','count_ret_1']
    Item_quantity = Item_quantity.merge(Temp,on='VisitNumber',how='left')

    Temp = New_DS[New_DS['ScanCount'] == -2 ].groupby(['VisitNumber']).count().Upc.reset_index()
    Temp.columns = ['VisitNumber','count_ret_2']
    Item_quantity = Item_quantity.merge(Temp,on='VisitNumber',how='left')

    Temp = New_DS[(New_DS['ScanCount'] < -2) & (New_DS['ScanCount'] >= -5) ].groupby(['VisitNumber']).count().Upc.reset_index()
    Temp.columns = ['VisitNumber','count_ret_2_5']
    Item_quantity = Item_quantity.merge(Temp,on='VisitNumber',how='left')

    Temp = New_DS[(New_DS['ScanCount'] < -5) & (New_DS['ScanCount'] >= -10) ].groupby(['VisitNumber']).count().Upc.reset_index()
    Temp.columns = ['VisitNumber','count_ret_5_10']
    Item_quantity = Item_quantity.merge(Temp,on='VisitNumber',how='left')

    Temp = New_DS[(New_DS['ScanCount'] < -10) ].groupby(['VisitNumber']).count().Upc.reset_index()
    Temp.columns = ['VisitNumber','count_ret_10']
    Item_quantity = Item_quantity.merge(Temp,on='VisitNumber',how='left')

    Item_quantity = Item_quantity.fillna(0)

    return Item_quantity
########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS):

    print("***************Starting Data cleansing***************")

    ##----------------------------------------------------------------------------------------------------------------##
    Train_DS = Train_DS.fillna(0)
    Actual_DS = Actual_DS.fillna(0)

    y = Train_DS.TripType.values
    Train_DS = Train_DS.drop(['TripType'], axis = 1)

    New_DS = pd.concat([Train_DS, Actual_DS])
    New_DS = New_DS.reset_index(drop=True).fillna(0)

    Upc_dummy = Get_Upc_dummy(New_DS)
    FN_dummy = Get_Fineline_dummy(New_DS)

    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(New_DS['DepartmentDescription']))
    New_DS['DepartmentDescription'] = lbl.transform(New_DS['DepartmentDescription'].astype(str))
    Train_DS['DepartmentDescription'] = lbl.transform(Train_DS['DepartmentDescription'].astype(str))

    DD_onehot_DS = DD_onehot_encoding(New_DS, Train_DS, y)
    Best_Trip_for_DD = Get_Best_Trip_for_DD(Train_DS,y)
    High_Lowest_Contributors = Get_High_Lowest_Contributors(New_DS)
    #Item_quantity = Get_Item_quantity(New_DS)

    High_Lowest_Contributors = pd.merge(High_Lowest_Contributors, DD_onehot_DS,on=['VisitNumber'],how='left')
    High_Lowest_Contributors = pd.merge(High_Lowest_Contributors, Best_Trip_for_DD,on=['DD_Max_per_visit_buy'],how='left')
    High_Lowest_Contributors = pd.merge(High_Lowest_Contributors, Upc_dummy,on=['VisitNumber'],how='left')
    High_Lowest_Contributors = pd.merge(High_Lowest_Contributors, FN_dummy,on=['VisitNumber'],how='left')

    print(np.shape(High_Lowest_Contributors))

    pd.DataFrame(High_Lowest_Contributors).to_csv(file_path+'High_Lowest_Contributors_cosine.csv')

    print("***************Ending Data cleansing***************")


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
    Train_DS      = pd.read_csv(file_path+'train.csv',sep=',')
    Actual_DS     = pd.read_csv(file_path+'test.csv',sep=',')

    Sample_DS     = pd.read_csv(file_path+'sample_submission.csv',sep=',')

    # #For testing only
    # Train_DS      = pd.read_csv(file_path+'train_100000.csv',sep=',', index_col=0,nrows = 8000 ).reset_index(drop=True)
    # Actual_DS     = pd.read_csv(file_path+'test_100000.csv',sep=',', index_col=0,nrows = 7000).reset_index(drop=True)

    Data_Munging(Train_DS,Actual_DS)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)