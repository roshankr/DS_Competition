import numpy as np
import scipy as sp
import sys
import pandas as pd # pandas
from time import time
from datetime import date,timedelta,datetime as dt
import datetime
import gc
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd # pandas
from sklearn.cross_validation import train_test_split
from random import *
from sklearn.preprocessing import Imputer
from sklearn.linear_model import  ElasticNet
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter

########################################################################################################################
# Walmart challenges participants to accurately predict the sales of 111 potentially weather-sensitive products
#  (like umbrellas, bread, and milk) around the time of major weather events at 45 of their retail locations.
########################################################################################################################

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS,Keys,Weather):
    print("***************Starting Data cleansing***************")

    #use imputer for missing values
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)
    cols = Weather.columns[2:42]
    cols2 =Weather.columns[0:2]
    Weather_New = Weather[cols]
    Weather_New = imp.fit_transform(Weather[cols])
    Weather_New2 = pd.concat([Weather[cols2],pd.DataFrame(Weather_New)],keys=None,axis=1)
    Weather_New2.columns = Weather.columns
    Weather = Weather_New2

    #Weather = Weather.drop(['date'], axis=1)

    Weather['SN'] = Weather['SN'].astype(float)
    Weather['SG'] = Weather['SG'].astype(float)
    Weather['RA'] = Weather['RA'].astype(float)
    Weather['snowfall'] = Weather['snowfall'].astype(float)
    Weather['preciptotal'] = Weather['preciptotal'].astype(float)

    Weather['snow']  = np.where ( (((Weather['SN'] == 1) | (Weather['SG'] == 1)) & (Weather['snowfall'] >= 2.0)),1,0)
    Weather['rain']  = np.where ( (((Weather['RA'] == 1) | (Weather['SN'] == 0)) & (Weather['preciptotal'] >= 1.0)),1,0)
    Weather['event']  = np.where ( (((Weather['snow'] == 1) | (Weather['rain'] == 1))),1,0)
    Weather['forecast'] = 0

    Weather = Weather.sort(['station_nbr', 'date'], ascending=[True,True])
    Weather = Weather.reset_index()

    for x in range(0, len(Weather)):

        if(Weather['event'][x] == 1):
            Weather['forecast'][x]   = 1
            Weather['forecast'][x-1] = 1
            Weather['forecast'][x-2] = 1
            Weather['forecast'][x-3] = 1
            Weather['forecast'][x+1] = 1
            Weather['forecast'][x+2] = 1
            Weather['forecast'][x+3] = 1

    Weather = Weather.sort(['index'], ascending=[True]).reset_index(drop=True).drop(['index'], axis=1)

    print(Weather.head(100))
    #Weather.to_csv(file_path+'Weather_new_val.csv')
    sys.exit(0)
    ####################################################################################################################
    #Reducign the samples for testing
    X_train, X_Sample, Y_train, Y_Sample = train_test_split(Train_DS, y, test_size=0.01, random_state=42)

    X_train, X_cv, Y_train, Y_cv = train_test_split(X_Sample, Y_Sample, test_size=0.2, random_state=42)

    print(X_train.head())
    print(X_cv.head())
    sys.exit(0)
    ####################################################################################################################

    Train_DS = Data_Merging(Train_DS,Keys,Weather)
    Actual_DS = Data_Merging(Actual_DS,Keys,Weather)

    # convert date to days
    # Train_DS["days"]  = ((pd.DatetimeIndex(Train_DS['date']).year)*365) + (pd.DatetimeIndex(Train_DS['date']).dayofyear)
    # Actual_DS["days"] = ((pd.DatetimeIndex(Actual_DS['date']).year)*365) +(pd.DatetimeIndex(Actual_DS['date']).dayofyear)
    # Train_DS = Train_DS.drop(['date','units'],axis=1)
    # Actual_DS = Actual_DS.drop('date',axis=1)

    Train_DS['date']  = Train_DS['date'].str.replace('-', '')
    Actual_DS['date'] = Actual_DS['date'].str.replace('-', '')

    y = Train_DS['units']
    Train_DS = Train_DS.drop('units',axis=1)

    #log transform the input files
    # Train_DS = np.log(1+Train_DS)
    # Actual_DS = np.log(1+Actual_DS)

    # Use PCA for feature extraction
    # pca = PCA(n_components=10)
    # pca.fit(Train_DS)
    # Train_DS = pca.transform(Train_DS)
    # X_CV = pca.transform(X_CV)
    # Actual_DS = pca.transform(Actual_DS)

    #Split into Training and Test sets
    X_train, X_cv, Y_train, Y_cv = train_test_split(Train_DS, y, test_size=0.4, random_state=42)

    # print("***************Ending Data cleansing***************")
    #
    return X_train, X_cv, Y_train, Y_cv,Actual_DS

########################################################################################################################
#Data reading and cleaning
########################################################################################################################
def Data_Merging(data,Keys,Weather):

    print("***************Starting Data Merging***************")

    data_keys = pd.merge(data, Keys)

    #data_with_wth = pd.merge(data_keys, Weather, left_on=['station_nbr', 'date'],
                            # right_on=['station_nbr', 'date'],how='inner')

    data_with_wth = pd.merge(data_keys, Weather, on=['station_nbr', 'date'],how='inner')

    print("***************Ending Data Merging***************")

    return data_with_wth

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
#Random Forest Regressor
########################################################################################################################
def RanFst_Regressor(X_train, X_cv, Y_train, Y_cv,Actual_DS,grid):

    print("***************Starting Random Forest Regressor***************")

    t0 = time()

    if grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "max_depth": [1, 2, 3, 4, 5, None],
                      "max_features": sp_randint(1, 11),
                      "min_samples_split": sp_randint(1, 11),
                      "min_samples_leaf": sp_randint(1, 11),
                      "bootstrap": [True, False]
                     }

        clf = RandomForestRegressor(n_estimators=1)

        # run randomized search
        n_iter_search = 20
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = 'mean_squared_error')

        start = time()
        clf.fit(X_train, Y_train)

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
        clf = RandomForestRegressor(n_estimators=1000)
        clf.fit(X_train, Y_train)

    preds = clf.predict(X_cv)
    print("CV Model predicted")

    score = np.sqrt(((np.log(preds+1) - np.log(Y_cv+1)) ** 2.0).mean())

    print("Random Forest Regressor - {0:.2f}".format(score))

    preds2 = clf.predict(Actual_DS)
    print("Actual Model predicted")

    print("***************Ending Random Forest Regressor***************")
    return preds2

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################

def main(argv):

    pd.set_option('display.width', 1000)
    pd.set_option('display.height', 1000)
    pd.options.mode.chained_assignment = None  # default='warn'
    gc.enable()

    global file_path
    file_path = 'C:/Python/Others/data/Kaggle/Walmart_Recruiting/'
    #file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Walmart_Recruiting/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################

    Train_DS =  pd.read_csv(file_path+'train.csv',sep=',')
    Actual_DS =  pd.read_csv(file_path+'test.csv',sep=',')
    Sample_DS = pd.read_csv(file_path+'sampleSubmission.csv',sep=',')
    Keys = pd.read_csv(file_path+'key.csv',sep=',')
    Weather = pd.read_csv(file_path+'weather 2.csv',sep=',')

    Train_DS = Train_DS[Train_DS['units'] > 0]

    #Train_DS.to_csv(file_path+'Train_DS_new_val.csv')

    Train_Summary = Train_DS.groupby(['store_nbr','item_nbr','date']).agg({'units': [np.mean]})

    Train_Summary.to_csv(file_path+'Train_Summary_new_val.csv')

    print(Train_Summary)
    sys.exit(0)

    X_train, X_cv, Y_train, Y_cv,Actual_DS = Data_Munging(Train_DS,Actual_DS,Keys,Weather)

    p_cv_RFC = RanFst_Regressor(X_train, X_cv, Y_train, Y_cv,Actual_DS,grid=False)

########################################################################################################################
#Get the predictions for actual data set
########################################################################################################################
    #Get the predictions for actual data set
    preds = pd.DataFrame(p_cv_RFC, index=Sample_DS.id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'Submission_Roshan.csv', index_label='id')

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)