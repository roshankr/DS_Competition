import requests
import numpy as np
import scipy as sp
import scipy.special as sps
import scipy.stats as st
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LogisticRegression
from scipy.stats import randint as sp_randint
from sklearn import decomposition, pipeline, metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
#import xgboost as xgb
#from sklearn.feature_extraction.text im
#from lasagne import layers
# from lasagne.nonlinearities import  softmax, rectify
# from lasagne.updates import nesterov_momentum,sgd,adagrad
# from lasagne.nonlinearities import identity
# from nolearn.lasagne import NeuralNetport TfidfVectorizer,TfidfTransformer
import collections
from sklearn.cross_validation import train_test_split
import random

########################################################################################################################
#Liberty Mutual Group: Property Inspection Prediction
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
def First_level_Stacking(X, y, X_Actual,crs_val,clfs):

    #(50999, 29)
    #(51000, 29)
    print("***************Starting First_level_Stacking*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    t0 = time()

    num_folds = 5

    #For storing the Train and test inputs for second level stacking
    DS_Blend_Train = np.zeros((X.shape[0], len(clfs)))
    DS_Blend_Test  = np.zeros((X_Actual.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):

        if j == 5:
            num_folds = 10
        else:
            num_folds = num_folds + 5

        crs_val = list(KFold(len(y), n_folds=num_folds,shuffle=False,indices=False))

        DS_Blend_Test_j = np.zeros((X_Actual.shape[0], len(crs_val)))

        scores= []

        print ("%d - Model is : %s" %(j+1 ,clf) )

        for i, (train, test) in enumerate(crs_val):

            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]

            clf.fit(X_train, np.log1p(y_train))
            y_pred = np.expm1(clf.predict(X_test))

            scores.append(normalized_gini(y_test,y_pred))

            print(" %d-iteration... %s " % (i+1,scores))

            DS_Blend_Train[test, j] = y_pred
            DS_Blend_Test_j[:, i] = clf.predict(X_Actual)

        DS_Blend_Test[:,j] = DS_Blend_Test_j.mean(1)
        scores=np.array(scores)
        print ("Normal CV Score:",np.mean(scores))
        print("-------------------------------------------------------------------------------------------------------")

    print("***************Ending First_level_Stacking*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    return DS_Blend_Train, DS_Blend_Test

########################################################################################################################
#Create models from multiple ensembling...
########################################################################################################################
def Get_Models_for_Stacking():

    print("***************Starting Get_Models_for_Stacking*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    t0 = time()

    clfs=[]

    random_list = random.sample(range(25), 10)

    # Get different Randowm Forest models
    for i in range(5):

        #i = random_list[j]

        if (np.isnan(Parm_RF_DS['max_depth'][i])):
            max_depth_val = None
        else:
            max_depth_val = int(Parm_RF_DS['max_depth'][i])

        clfs.append(RandomForestRegressor(n_estimators=1000
                                         ,min_samples_leaf=Parm_RF_DS['min_samples_leaf'][i]
                                         ,max_features=Parm_RF_DS['max_features'][i]
                                         ,bootstrap=Parm_RF_DS['bootstrap'][i]
                                         ,min_samples_split=Parm_RF_DS['min_samples_split'][i]
                                         ,max_depth=max_depth_val
                                         ,n_jobs=-1))

    ####################################################################################################################
    # Get different XGB models
    for i in range(5):

        #i = random_list[j]

        clfs.append(xgb.XGBRegressor(n_estimators=1500
                                    ,max_depth=Parm_XGB_DS['max_depth'][i]
                                    ,learning_rate=0.01
                                    ,nthread=4
                                    ,min_child_weight=Parm_XGB_DS['min_child_weight'][i]
                                    ,subsample=Parm_XGB_DS['subsample'][i]
                                    ,colsample_bytree=Parm_XGB_DS['colsample_bytree'][i]
                                    ,silent=Parm_XGB_DS['silent'][i]
                                    ,gamma=Parm_XGB_DS['gamma'][i]))


    print("***************Ending Get_Models_for_Stacking*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    return clfs

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def Stack_with_nfold(Train_DS, y , Actual_DS,Train_DS_Stack,Actual_DS_Stack,Train_DS_CVR):

    print("***************Starting Stack_with_nfold*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    t0 = time()

    Train_DS  = np.array(Train_DS)
    Actual_DS = np.array(Actual_DS)
    y         = np.array(y)

    #define CV plan
    crs_val = list(KFold(len(y), n_folds=num_folds,shuffle=False,indices=False))

    #Define the models

    # clfs = [RandomForestRegressor(n_estimators=2000,min_samples_leaf=19,max_features=12,bootstrap=True,min_samples_split=1,max_depth=25,n_jobs=-1)
    #        ,ExtraTreesRegressor(n_estimators=1000,min_samples_leaf=19,max_features=12,bootstrap=True,min_samples_split=1,max_depth=25,n_jobs=-1)
    #        ,GradientBoostingRegressor(n_estimators=1000)
    #        ,xgb.XGBRegressor(n_estimators=2000,max_depth=5,learning_rate=0.01,nthread=4,min_child_weight=19,subsample=1,colsample_bytree=0.3,silent=True,gamma=0.6)
    #        ,SVR(max_iter=-1)
    #        ]

    clfs = [ExtraTreesRegressor(n_estimators=200,min_samples_leaf=19,max_features=12,bootstrap=True,min_samples_split=1,max_depth=25,n_jobs=-1)
           ]

    #clfs = Get_Models_for_Stacking()

    Train_DS, Actual_DS = First_level_Stacking(Train_DS, y, Actual_DS,crs_val,clfs)

    #cols = Train_DS_CVR.columns
    #Train_DS_CVR = Train_DS_CVR.append(pd.DataFrame(DS_CV_Result,columns=cols),ignore_index=True)

    #Take a backup of Train DS Actual DS , in case if we want to run Final stacking alone
    Train_DS1 = pd.DataFrame(Train_DS)
    #Train_DS1['y'] = y
    Actual_DS1 = pd.DataFrame(Actual_DS)

    Train_DS1 = pd.concat([Train_DS_Stack,Train_DS1],axis=1).reset_index(drop=True)
    Actual_DS1 = pd.concat([Actual_DS_Stack,Actual_DS1],axis=1).reset_index(drop=True)

    if Write_first_stack:
        pd.DataFrame(Train_DS1).to_csv(file_path+'Train_DS_Stacking_Nfold_v3.csv')
        pd.DataFrame(Actual_DS1).to_csv(file_path+'Actual_DS_Stacking_Nfold_v3.csv')
        #pd.DataFrame(Train_DS_CVR).to_csv(file_path+'First_level_CV_Result_v3.csv')

    print("***************Ending Stack_with_nfold*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    return Train_DS, Actual_DS

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, gini_scorer,Parm_RF_DS, Parm_XGB_DS, num_folds, Write_first_stack

    num_folds = 5
    Write_first_stack = False
    # Normalized Gini Scorer
    gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)

    if(platform.system() == "Windows"):
        file_path = 'C:/Python/Others/data/Kaggle/Liberty_Mutual_Group/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Liberty_Mutual_Group/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Sample_DS       = pd.read_csv(file_path+'sample_submission.csv',sep=',')
    Parm_RF_DS      = pd.read_csv(file_path+'Parms_DS_RF2.csv',sep=',', index_col=0)
    Parm_XGB_DS     = pd.read_csv(file_path+'Parms_DS_XGB_1000_2.csv',sep=',', index_col=0)
    Train_DS        = pd.read_csv(file_path+'train.csv',sep=',', index_col=0)
    Actual_DS       = pd.read_csv(file_path+'test.csv',sep=',', index_col=0)
    Train_DS_Stack  = pd.read_csv(file_path+'Train_DS_Stacking_Nfold_v3_reduced.csv',sep=',', index_col=0)
    Actual_DS_Stack = pd.read_csv(file_path+'Actual_DS_Stacking_Nfold_v3_reduced.csv',sep=',', index_col=0)
    Train_DS_CVR    = pd.read_csv(file_path+'First_level_CV_Result_v3.csv',sep=',', index_col=0).reset_index(drop=True)

    Train_DS, Actual_DS, y = Data_Munging(Train_DS,Actual_DS)

    Train_DS, Actual_DS = Stack_with_nfold(Train_DS, y , Actual_DS,Train_DS_Stack,Actual_DS_Stack,Train_DS_CVR)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)