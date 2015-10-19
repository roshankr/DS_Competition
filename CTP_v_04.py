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
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint as sp_randint
from sklearn import decomposition, pipeline, metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.feature_extraction import *
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.feature_extraction import DictVectorizer

#import xgboost as xgb

########################################################################################################################
#Caterpillar Tube Pricing
########################################################################################################################
#--------------------------------------------Algorithm : Random Forest :------------------------------------------------
#Random Forest :
#1. First of all set up a good CV plan . Understood Kfold with 10 folds , with unique Assem Id would be the most closer
#2. Run RF with only raw Train data CV: 0.38442, no feature eng required
#--------------------------------------------Algorithm : XGB------------------------------------------------------------
#XGB :

#--------------------------------------------Suggestions, Ideas---------------------------------------------------------
#Suggestions, Ideas
#1. As per the grid output , better to run grid search CV=10 and max possible iterations >100 - TODO
#2. Try doing feature engineering - TODO
#3. Try Graphlab GBM - TODO
#4. Try Vowpal Wobit - TODO

########################################################################################################################
#RMSLE Scorer
########################################################################################################################
def RMSLE(solution, submission):
    assert len(solution) == len(submission)
    score = np.sqrt(((np.log(solution+1) - np.log(submission+1)) ** 2.0).mean())
    return score

########################################################################################################################
#Utility function to report best scores
########################################################################################################################
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

########################################################################################################################
#Cross Validation and model fitting , using uniqueassembly id as split between testCV and trainCV
########################################################################################################################
def Nfold_Cross_Valid_New(X, y, clf):

    print("***************Starting Kfold Cross validation***************")

    #X =np.array(X)
    scores=[]
    unique_labels = np.unique(X[:,0])

    X = pd.DataFrame(X)
    X['y'] = y


    #ss = StratifiedShuffleSplit(y, n_iter=10,test_size=0.3, random_state=42, indices=None)
    ss = KFold(len(unique_labels), n_folds=10,shuffle=True,indices=False)
    i = 1

    for trainCV, testCV in ss:

        test_labels = unique_labels[testCV]

        X_test   = X[X[0].isin(test_labels)]
        y_test   = X_test['y']
        X_test   = X_test.drop(['y',0], axis = 1)

        X_train  = X[~X[0].isin(test_labels)]
        y_train  = X_train['y']
        X_train  = X_train.drop(['y',0], axis = 1)

        print(np.shape(X_test))
        print(np.shape(X_train))

        y_train = np.log1p(y_train)
        clf.fit(X_train, y_train)
        y_pred=np.expm1(clf.predict(X_test))
        scores.append(RMSLE(y_test,y_pred))
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
        y_train = np.log1p(y_train)
        clf.fit(X_train, y_train)
        y_pred=np.expm1(clf.predict(X_test))
        scores.append(RMSLE(y_test,y_pred))
        print(" %d-iteration... %s " % (i,scores))
        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Get Tube volume
########################################################################################################################
def Tube_Volume(row):
    Radius =  (row['diameter'] - row['wall'])/2
    Volume = (3.14*Radius*Radius*row['length'])/1000
    return Volume

########################################################################################################################
#Get Tube Area
########################################################################################################################
def Tube_Area(row):
    Radius =  (row['diameter'] - row['wall'])/2
    Area = ((2*3.14*Radius*Radius) + (2*3.14*Radius*row['length']))/1000
    return Area

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS,Tube_DS,Bill_DS,Spec_DS,Tube_End_DS, Comp_DS):

    print("***************Starting Data cleansing***************")
    global col_vals

    y = Train_DS.cost.values

    #Get month and Day from Date feature
    temp = pd.DatetimeIndex(Train_DS['quote_date'])
    Train_DS['month']  = temp.month
    Train_DS['year']   = temp.year
    Train_DS['dayofyear'] = temp.dayofyear
    Train_DS['weekofyear'] = temp.weekofyear
    Train_DS['dayofweek'] = temp.dayofweek

    temp = pd.DatetimeIndex(Actual_DS['quote_date'])
    Actual_DS['month']  = temp.month
    Actual_DS['year']   = temp.year
    Actual_DS['dayofyear'] = temp.dayofyear
    Actual_DS['weekofyear'] = temp.weekofyear
    Actual_DS['dayofweek'] = temp.dayofweek

    Train_DS  = Train_DS.drop(['cost','quote_date'], axis = 1)
    Actual_DS = Actual_DS.drop(['id','quote_date'], axis = 1)

    ####################################################################################################################
    #Clean Tube_DS
    Tube_DS['material_id'] = Tube_DS['material_id'].fillna('SP-9999')
    Tube_DS['end_a'] = Tube_DS['end_a'].replace('NONE','9999')
    Tube_DS['end_x'] = Tube_DS['end_x'].replace('NONE','9999')

    #Merge Tubes with Tube end
    Tube_DS = pd.merge(Tube_DS,Tube_End_DS,left_on=['end_a'],right_on=['end_form_id'],how='left')
    Tube_DS  = Tube_DS.drop(['end_form_id'], axis = 1)

    Tube_DS = pd.merge(Tube_DS,Tube_End_DS,left_on=['end_x'],right_on=['end_form_id'],how='left')
    Tube_DS  = Tube_DS.drop(['end_form_id'], axis = 1)

    Tube_DS['forming_x'] = Tube_DS['forming_x'].fillna('NONE')
    Tube_DS['forming_y'] = Tube_DS['forming_y'].fillna('NONE')

    Tube_DS['volume']  = Tube_DS.apply(Tube_Volume, axis=1)
    Tube_DS['Area']    = Tube_DS.apply(Tube_Area, axis=1)

    ####################################################################################################################
    #Clean Component DS
    Comp_DS = Comp_DS.fillna(0)
    Comp_DS  = Comp_DS.drop(['name'], axis = 1)
    Comp_DS['component_type_id'].replace('OTHER','CP-999', regex=True, inplace= True)
    Comp_Unique = list(pd.unique(Comp_DS['component_type_id'].values.ravel()))
    Comp_Unique.sort()

    ####################################################################################################################
    #Clean Bill_DS
    for i in range(1,9):
        column_label = 'component_id_'+str(i)
        Bill_DS[column_label].replace(np.nan,'C-0000', regex=True, inplace= True)
        #Bill_DS[column_label] = Bill_DS[column_label].str.replace('C-','').astype(float)

    Bill_DS = Bill_DS.fillna(0)

    Bill_DS_New2 = pd.DataFrame(columns=['weight'], index=Bill_DS['tube_assembly_id'] )

    row_iterator = Bill_DS.iterrows()
    print("starting Bill iteration")
    for i, row in row_iterator:
        row_val = row['tube_assembly_id']
        Comp_Weight = 0

        for j in range(1,9):
            column_label    = 'component_id_'+str(j)
            quantity_label = 'quantity_'+str(j)
            if row[column_label] !='C-0000':
                col_val = row[column_label]
                Comp_Type = (Comp_DS[Comp_DS['component_id']==col_val]['weight']).max()
                Comp_Weight = Comp_Weight + (row[quantity_label] * Comp_Type)

        Bill_DS_New2.loc[row_val]['weight'] = Comp_Weight

    Bill_DS = Bill_DS_New2.reset_index()
    print(Bill_DS.head())

    ####################################################################################################################
    #Clean Spec_DS
    # Spec_DS = Spec_DS.fillna(0)
    #
    # Spec_DS_New = Spec_DS.drop(['tube_assembly_id'], axis = 1)
    # Spec_Unique = list(pd.unique(Spec_DS_New.values.ravel()))
    # Spec_Unique = [x for x in Spec_Unique if str(x) != '0']
    # Spec_DS_New2 = pd.DataFrame(columns=Spec_Unique, index=Spec_DS['tube_assembly_id'] )
    #
    # row_iterator = Spec_DS.iterrows()
    # print("starting Spec iteration")
    # for i, row in row_iterator:
    #     row_val = row['tube_assembly_id']
    #     Spec_DS_New2.loc[row_val] = 0
    #
    #     for j in range(1,11):
    #         column_label    = 'spec'+str(j)
    #
    #         if row[column_label] !=0:
    #             col_val = row[column_label]
    #             Spec_DS_New2.loc[row_val][col_val] = 1
    #
    # print(Spec_DS_New2.head(20))
    #
    # #Spec_DS = Spec_DS_New2.reset_index()
    # Spec_DS = Spec_DS_New2
    #print(Spec_DS.head())
    ####################################################################################################################
    # Get non zero specs for each row
    Spec_DS['sp_count'] = Spec_DS.count(axis=1) - 1

    for i in range(1,11):
        column_label = 'spec'+str(i)
        Spec_DS[column_label].replace(np.nan,'SP-0000', regex=True, inplace= True)
        #Spec_DS[column_label] = Spec_DS[column_label].str.replace('SP-','').astype(int)

    #Spec_DS  = Spec_DS.drop(['spec9','spec10'], axis = 1)
    ####################################################################################################################
    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    #Merge Train , Actual with Tubes & Bills
    Train_DS  = pd.merge(Train_DS,Tube_DS,on='tube_assembly_id',how='inner')
    Train_DS  = pd.merge(Train_DS,Bill_DS,on='tube_assembly_id',how='inner')
    Train_DS  = pd.merge(Train_DS,Spec_DS,on='tube_assembly_id',how='inner')

    Actual_DS = pd.merge(Actual_DS,Tube_DS,on='tube_assembly_id',how='inner')
    Actual_DS = pd.merge(Actual_DS,Bill_DS,on='tube_assembly_id',how='inner')
    Actual_DS = pd.merge(Actual_DS,Spec_DS,on='tube_assembly_id',how='inner')

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

####################################################################################################################
    # vec = DictVectorizer(sparse=False)
    # x_dv = vec.fit_transform((Train_DS.ix[:,'supplier':].append(Actual_DS.ix[:,'supplier':])).reset_index(drop=True).T.to_dict().values())
    # Train_DS_Dict  = x_dv[:len(Train_DS), :]
    # Actual_DS_Dict = x_dv[len(Train_DS):, :]
    #
    # Train_DS  = np.append(Train_DS['tube_assembly_id'], Train_DS_Dict ,1)
    # Actual_DS = np.append(Actual_DS['tube_assembly_id'], Actual_DS_Dict ,1)
    #
    # print(Train_DS.head())

####################################################################################################################

    #Get col types for Train
    col_types = (Train_DS.dtypes).reset_index(drop=True)
    col_vals = list(Train_DS.columns)
    del col_vals[0]
    col_vals = pd.DataFrame(col_vals)

    Train_DS = Train_DS.fillna(0)
    Actual_DS = Actual_DS.fillna(0)

    Train_DS = np.array(Train_DS)
    Actual_DS = np.array(Actual_DS)

    ####################################################################################################################
    print("Starting label encoding")
    # Convert categorical data to numbers
    for i in range(Train_DS.shape[1]):
        #if i in [0,3,8,14,15,16,17,18,19,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]:
        if col_types[i] =='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(Train_DS[:,i]) + list(Actual_DS[:,i]))
            Train_DS[:,i] = lbl.transform(Train_DS[:,i])
            Actual_DS[:,i] = lbl.transform(Actual_DS[:,i])

    ####################################################################################################################
    # print("Starting log transforming")
    Train_DS = np.log(1+np.asarray(Train_DS, dtype=np.float32))
    Actual_DS = np.log(1+np.asarray(Actual_DS, dtype=np.float32))

    #Setting Standard scaler for data
    #stdScaler = StandardScaler()
    #stdScaler.fit(Train_DS,y)
    #Train_DS = stdScaler.transform(Train_DS)
    #Actual_DS = stdScaler.transform(Actual_DS)

    print("***************Ending Data cleansing***************")

    return Train_DS, Actual_DS, y

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def RFR_Regressor(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting Random Forest Regressor***************")
    t0 = time()

    if grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "max_depth": [1, 3, 5,8,10,12,15,20,25,30, None],
                      "max_features": sp_randint(1, 49),
                      "min_samples_split": sp_randint(1, 49),
                      "min_samples_leaf": sp_randint(1, 49),
                      "bootstrap": [True, False]
                     }

        clf = RandomForestRegressor(n_estimators=100)

        # run randomized search
        n_iter_search = 25
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = RMSLE_scorer,cv=10,n_jobs=2)

        #Remove tube_assembly_id after its been used in cross validation
        Train_DS    = np.delete(Train_DS,0, axis = 1)

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

        #for testing purpose
        #New CV:0.2721 ,  LB: 0.266438 (with all features)
        clf = RandomForestRegressor(n_estimators=200)

        Nfold_score = Nfold_Cross_Valid_New(Train_DS, y, clf)

        #Remove tube_assembly_id after its been used in cross validation
        Train_DS    = np.delete(Train_DS,0, axis = 1)
        Actual_DS   = np.delete(Actual_DS,0, axis = 1)

        Train_DS, y = shuffle(Train_DS, y, random_state=42)
        y = np.log1p(y)

        clf.fit(Train_DS, y)

        feature = pd.DataFrame()
        feature['imp'] = clf.feature_importances_
        feature['col'] = col_vals
        feature = feature.sort(['imp'], ascending=False)
        print(feature)

    #Predict actual model
    pred_Actual = np.expm1(clf.predict(Actual_DS))
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_RF_2.csv', index_label='id')

    print("***************Ending Random Forest Regressor***************")
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, RMSLE_scorer

    # RMSLE_scorer
    RMSLE_scorer = metrics.make_scorer(RMSLE, greater_is_better = False)

    if(platform.system() == "Windows"):
        file_path = 'C:/Python/Others/data/Kaggle/Caterpillar_Tube_Pricing/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Caterpillar_Tube_Pricing/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS      = pd.read_csv(file_path+'competition_data/train_set.csv',sep=',')
    Actual_DS     = pd.read_csv(file_path+'competition_data/test_set.csv',sep=',')
    Tube_DS       = pd.read_csv(file_path+'competition_data/tube.csv',sep=',')
    Bill_DS       = pd.read_csv(file_path+'competition_data/bill_of_materials.csv',sep=',')
    Spec_DS       = pd.read_csv(file_path+'competition_data/specs.csv',sep=',')
    Tube_End_DS   = pd.read_csv(file_path+'competition_data/tube_end_form.csv',sep=',')
    Comp_DS       = pd.read_csv(file_path+'competition_data/components_2.csv',sep=',')
    Sample_DS     = pd.read_csv(file_path+'sample_submission.csv',sep=',')


    Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS,Tube_DS,Bill_DS,Spec_DS,Tube_End_DS, Comp_DS)

    pred_Actual = RFR_Regressor(Train_DS, y, Actual_DS, Sample_DS, grid=False)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)
