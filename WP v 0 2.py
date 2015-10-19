import numpy as np
import scipy as sp
import sys
import pandas as pd # pandas
from time import time
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

########################################################################################################################
# Walmart challenges participants to accurately predict the sales of 111 potentially weather-sensitive products
#  (like umbrellas, bread, and milk) around the time of major weather events at 45 of their retail locations.
########################################################################################################################

########################################################################################################################
#Data reading and cleaning
########################################################################################################################
def Data_Munging(data,Keys,Weather):

    print("***************Starting Data clean up***************")

    data_keys = pd.merge(data, Keys)

    #data_with_wth = pd.merge(data_keys, Weather, left_on=['station_nbr', 'date'], right_on=['station_nbr', 'date'],how='inner')
    data_with_wth = pd.merge(data_keys, Weather, on=['station_nbr', 'date'],how='inner')

    print(data_with_wth.head())
    print("***************Ending Data clean up***************")

    return data_with_wth

########################################################################################################################
#Random Forest Regressor
########################################################################################################################
def RanFst_Regressor(X_train, X_CV, Y_train,Y_CV):

    print("***************Starting Random Forest Regressor***************")

    t0 = time()
    print(np.shape(X_train))
    print(np.shape(X_CV))
    print(np.shape(Y_train))
    print(np.shape(Y_CV))

    clf = RandomForestRegressor(n_estimators=100,n_jobs=-1)

    #clf = SVC(kernel='rbf', class_weight='auto',C=1e5, gamma= 0.01,probability=True)

    clf.fit(X_train, Y_train)

    preds = clf.predict(X_CV)
    score = clf.score(X_CV,Y_CV)
    print("Random Forest Classifier - {0:.2f}%".format(100 * score))

    print("***************Ending Random Forest Regressor***************")
    return score

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################

def main(argv):

    pd.set_option('display.width', 500)
    pd.set_option('display.height', 500)
    pd.options.mode.chained_assignment = None  # default='warn'
    gc.enable()


########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

    #train =  pd.read_csv('C:/Python/Others/data/Kaggle/Walmart_Recruiting/train.csv',sep=',')
    #actual =  pd.read_csv('C:/Python/Others/data/Kaggle/Walmart_Recruiting/test.csv',sep=',')
    #Sample_DS = pd.read_csv('C:/Python/Others/data/Kaggle/Walmart_Recruiting/sampleSubmission.csv',sep=',')
    Keys = pd.read_csv('C:/Python/Others/data/Kaggle/Walmart_Recruiting/key.csv',sep=',')
    Weather = pd.read_csv('C:/Python/Others/data/Kaggle/Walmart_Recruiting/weather 3.csv',sep=',')

    train =  pd.read_csv('C:/Python/Others/data/Kaggle/Walmart_Recruiting/X_train.csv',sep=',')
    CV =  pd.read_csv('C:/Python/Others/data/Kaggle/Walmart_Recruiting/X_CV.csv',sep=',')

    Y_train = train.units.values
    Y_CV    = CV.units.values
    X_train = train.drop(['units'], axis=1)
    X_CV    = CV.drop(['units'], axis=1)

    print(np.shape(X_train))
    print(np.shape(X_CV))

    p_cv_RFC = RanFst_Regressor(X_train, X_CV, Y_train,Y_CV)
########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)