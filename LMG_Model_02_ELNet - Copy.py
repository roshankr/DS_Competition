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

    for i in range(Train_DS.shape[1]):
        if col_types[i]  != 'object':
           Train_Dict_DS = np.delete(Train_Dict_DS, i, 1)
           Actual_Dict_DS = np.delete(Actual_Dict_DS, i, 1)

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

    ####################################################################################################################
    #Get some new features


    Train_DS = np.append(Train_DS, np.amax(Train_DS,axis=1).reshape(-1,1),1)
    Train_DS = np.append(Train_DS, np.sum(Train_DS,axis=1).reshape(-1,1),1)

    Actual_DS = np.append(Actual_DS, np.amax(Actual_DS,axis=1).reshape(-1,1),1)
    Actual_DS = np.append(Actual_DS, np.sum(Actual_DS,axis=1).reshape(-1,1),1)

    ####################################################################################################################
    #Merge De-vectorizer
    Train_DS = np.append(Train_DS,Train_Dict_DS,1)
    Actual_DS = np.append(Actual_DS,Actual_Dict_DS,1)

    Train_DS = Train_DS.astype(float)
    Actual_DS = Actual_DS.astype(float)

    ####################################################################################################################
    print("starting TFID conversion...")
    tfv = TfidfTransformer()
    tfv.fit(Train_DS)
    Train_DS1 =  tfv.transform(Train_DS).toarray()
    Actual_DS1 = tfv.transform(Actual_DS).toarray()

    Train_DS = np.append(Train_DS,Train_DS1,1)
    Actual_DS = np.append(Actual_DS,Actual_DS1,1)

    Train_DS, y = shuffle(Train_DS, y, random_state=21)

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

    # Used to store the CV results for each fold (including model and cv avg)
    DS_CV_Result = np.empty((len(clfs),num_folds+2 ),dtype=object)

    #For storing the Train and test inputs for second level stacking
    DS_Blend_Train = np.zeros((X.shape[0], len(clfs)))
    DS_Blend_Test  = np.zeros((X_Actual.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):

        DS_Blend_Test_j = np.zeros((X_Actual.shape[0], len(crs_val)))

        scores= []

        DS_CV_Result[j,0] = clf

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

            DS_CV_Result[j,i+1] = normalized_gini(y_test,y_pred)

        DS_CV_Result[j,i+2] = np.mean(scores)
        DS_Blend_Test[:,j] = DS_Blend_Test_j.mean(1)
        scores=np.array(scores)
        print ("Normal CV Score:",np.mean(scores))
        print("-------------------------------------------------------------------------------------------------------")

    print("***************Ending First_level_Stacking*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    return DS_Blend_Train, DS_Blend_Test, DS_CV_Result

########################################################################################################################
#Create models from multiple ensembling...
########################################################################################################################
def Get_Models_for_Stacking():

    print("***************Starting Get_Models_for_Stacking*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    t0 = time()

    clfs=[]

    random_list = random.sample(range(25), 10)

    # Get different Randowm Forest models
    for j in range(3):

        i = random_list[j]

        if (np.isnan(Parm_RF_DS['max_depth'][i])):
            max_depth_val = None
        else:
            max_depth_val = int(Parm_RF_DS['max_depth'][i])

        clfs.append(RandomForestRegressor(n_estimators=100
                                         ,min_samples_leaf=Parm_RF_DS['min_samples_leaf'][i]
                                         ,max_features=None
                                         ,bootstrap=Parm_RF_DS['bootstrap'][i]
                                         ,min_samples_split=Parm_RF_DS['min_samples_split'][i]
                                         ,max_depth=max_depth_val
                                         ,n_jobs=2))

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

    clfs = [ElasticNet(alpha=0.1,   l1_ratio=0.1)
           ,ElasticNet(alpha=0.01,  l1_ratio=0.1)
           ,ElasticNet(alpha=0.002,  l1_ratio=0.1)
           ]


    Train_DS, Actual_DS, DS_CV_Result = First_level_Stacking(Train_DS, y, Actual_DS,crs_val,clfs)

    cols = Train_DS_CVR.columns
    Train_DS_CVR = Train_DS_CVR.append(pd.DataFrame(DS_CV_Result,columns=cols),ignore_index=True)

    #Take a backup of Train DS Actual DS , in case if we want to run Final stacking alone
    Train_DS1 = pd.DataFrame(Train_DS)
    #Train_DS1['y'] = y
    Actual_DS1 = pd.DataFrame(Actual_DS)

    Train_DS1 = pd.concat([Train_DS_Stack,Train_DS1],axis=1).reset_index(drop=True)
    Actual_DS1 = pd.concat([Actual_DS_Stack,Actual_DS1],axis=1).reset_index(drop=True)

    if Write_first_stack:
        pd.DataFrame(Train_DS1).to_csv(file_path+'Train_DS_Stacking_Nfold_v2.csv')
        pd.DataFrame(Actual_DS1).to_csv(file_path+'Actual_DS_Stacking_Nfold_v2.csv')
        pd.DataFrame(Train_DS_CVR).to_csv(file_path+'First_level_CV_Result_v2.csv')

    print("***************Ending Stack_with_nfold*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    return Train_DS, Actual_DS

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, gini_scorer,Parm_RF_DS, Parm_XGB_DSmn, num_folds, Write_first_stack

    num_folds = 5
    Write_first_stack = True

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
    Train_DS_Stack  = pd.read_csv(file_path+'Train_DS_Stacking_Nfold_v2.csv',sep=',', index_col=0)
    Actual_DS_Stack = pd.read_csv(file_path+'Actual_DS_Stacking_Nfold_v2.csv',sep=',', index_col=0)
    Train_DS_CVR    = pd.read_csv(file_path+'First_level_CV_Result_v2.csv',sep=',', index_col=0).reset_index(drop=True)

    Train_DS, Actual_DS, y = Data_Munging(Train_DS,Actual_DS)

    Train_DS, Actual_DS = Stack_with_nfold(Train_DS, y , Actual_DS,Train_DS_Stack,Actual_DS_Stack,Train_DS_CVR)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)