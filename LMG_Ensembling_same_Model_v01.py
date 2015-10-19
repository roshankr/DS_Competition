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
import time as tm
import warnings
from math import sqrt, exp, log
from csv import DictReader
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer ,TfidfTransformer
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

########################################################################################################################
#Liberty Mutual Group: Property Inspection Prediction
########################################################################################################################
# This program will use teh best parms for each model and ensemble it in multiple ways to see the best one
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
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid(X, y, ss , clf):

    print("***************Starting Kfold Cross validation*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    X =np.array(X)

    scores=[]
    DS_Blend_Train = np.zeros((X.shape[0], 1))

    #ss = StratifiedShuffleSplit(y, n_iter=10,test_size=0.3, random_state=42, indices=None)
    #ss = KFold(len(y), n_folds=10,shuffle=True,indices=False)
    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)

        DS_Blend_Train[testCV, 0] = y_pred

        scores.append(normalized_gini(y_test,y_pred))
        print(" %d-iteration... %s " % (i,scores))
        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation*************** at Time: %s" %(tm.strftime("%H:%M:%S")))

    return DS_Blend_Train

########################################################################################################################
#Create models from multiple ensembling...
########################################################################################################################
def Get_Models_for_Stacking():

    print("***************Starting Get_Models_for_Ensembling*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    t0 = time()

    clfs=[]

    # Get different Random Forest models
    for i in range(2):

        if (np.isnan(Parm_RF_DS['max_depth'][i])):
            max_depth_val = None
        else:
            max_depth_val = int(Parm_RF_DS['max_depth'][i])

        clfs.append(RandomForestRegressor(n_estimators=200
                                         ,min_samples_leaf=Parm_RF_DS['min_samples_leaf'][i]
                                         ,max_features=None
                                         ,bootstrap=Parm_RF_DS['bootstrap'][i]
                                         ,min_samples_split=Parm_RF_DS['min_samples_split'][i]
                                         ,max_depth=max_depth_val
                                         ,n_jobs=-1))

    print("***************Ending Get_Models_for_Ensembling*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    return clfs

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS):

    print("***************Starting Data cleansing*************** at Time: %s" %(tm.strftime("%H:%M:%S")))

    y = Train_DS.Hazard.values

    Train_DS = Train_DS.drop(['Hazard'], axis = 1)

    Train_DS  = Train_DS.drop(['T2_V10','T2_V7','T1_V13','T1_V10'], axis = 1)
    Actual_DS = Actual_DS.drop(['T2_V10','T2_V7','T1_V13','T1_V10'], axis = 1)

    # global columns
    columns = Train_DS.columns
    col_types = (Train_DS.dtypes).reset_index(drop=True)

    ####################################################################################################################
    #perform De-vectorizer
    Train_Dict_DS = Train_DS.T.to_dict().values()
    Actual_Dict_DS = Actual_DS.T.to_dict().values()

    vec = DictVectorizer(sparse=False)
    Train_Dict_DS = vec.fit_transform(Train_Dict_DS)
    Actual_Dict_DS = vec.transform(Actual_Dict_DS)

    # for i in range(Train_DS.shape[1]):
    #     if col_types[i]  != 'object':
    #        Train_Dict_DS = np.delete(Train_Dict_DS, i, 1)
    #        Actual_Dict_DS = np.delete(Actual_Dict_DS, i, 1)

    ####################################################################################################################

    Train_DS = np.array(Train_DS)
    Actual_DS = np.array(Actual_DS)

    print("Starting label encoding")
    # label encode the categorical variables
    for i in range(Train_DS.shape[1]):
        if col_types[i] =='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(Train_DS[:,i]) + list(Actual_DS[:,i]))
            Train_DS[:,i] = lbl.transform(Train_DS[:,i])
            Actual_DS[:,i] = lbl.transform(Actual_DS[:,i])

    ####################################################################################################################
    #Merge De-vectorizer
    Train_DS = np.append(Train_DS,Train_Dict_DS,1)
    Actual_DS = np.append(Actual_DS,Actual_Dict_DS,1)

    Train_DS = Train_DS.astype(float)
    Actual_DS = Actual_DS.astype(float)

    print("starting TFID conversion...")
    tfv = TfidfTransformer()
    tfv.fit(Train_DS)
    Train_DS1 =  tfv.transform(Train_DS).toarray()
    Actual_DS1 = tfv.transform(Actual_DS).toarray()

    Train_DS = np.append(Train_DS,Train_DS1,1)
    Actual_DS = np.append(Actual_DS,Actual_DS1,1)

    Train_DS = np.log( 1 + Train_DS)
    Actual_DS = np.log( 1 + Actual_DS)

    #Setting Standard scaler for data
    stdScaler = StandardScaler()
    stdScaler.fit(Train_DS,y)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

    Train_DS, y = shuffle(Train_DS, y, random_state=21)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    print("***************Ending Data cleansing*************** at Time: %s" %(tm.strftime("%H:%M:%S")))

    return Train_DS, y, Actual_DS

########################################################################################################################
#Ensemble with fold
########################################################################################################################
def Ensemble_with_nfold(Train_DS, y , Actual_DS, Sample_DS):

    print("***************Starting Ensemble_with_nfold*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    t0 = time()

    #define CV plan
    ss = list(KFold(len(y), n_folds=5,shuffle=True,indices=False))

    clfs = Get_Models_for_Stacking()

    scores = []
    Ensemble_DS_output  = np.zeros((Actual_DS.shape[0], len(clfs)))
    Ensemble_DS_Train   = np.zeros((Train_DS.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):

        print(" Training Model...%d" % (j+1))

        Ensemble_DS_Train[:,j]  = Nfold_Cross_Valid(Train_DS, y, ss , clf)

        #Fit and predict the model
        clf.fit(Train_DS, y)
        Ensemble_DS_output[:,j] = np.array(clf.predict(Actual_DS))

    #Do what ever Ensembling you want
    pred_Train  = Ensemble_DS_Train.mean(1)
    Ens_score = normalized_gini(y,pred_Train)
    print(" Ensemble output of Train is... %d " % (Ens_score))

    pred_Actual = Ensemble_DS_output.mean(1)

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Ensembling.csv', index_label='Id')

    print("***************Ending Ensemble_with_nfold*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    return Ensemble_DS_output

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, gini_scorer,Parm_RF_DS, Parm_XGB_DS

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
    Sample_DS   = pd.read_csv(file_path+'sample_submission.csv',sep=',')
    Parm_RF_DS  = pd.read_csv(file_path+'Parms_DS_RF2.csv',sep=',', index_col=0)
    Parm_XGB_DS = pd.read_csv(file_path+'Parms_DS_XGB_1000_2.csv',sep=',', index_col=0)
    Train_DS    =  pd.read_csv(file_path+'train.csv',sep=',', index_col=0)
    Actual_DS   =  pd.read_csv(file_path+'test.csv',sep=',', index_col=0)

    Train_DS, y, Actual_DS = Data_Munging(Train_DS,Actual_DS)

    Ensemble_DS_output = Ensemble_with_nfold(Train_DS, y , Actual_DS, Sample_DS)
########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)