import requests
import numpy as np
import scipy as sp
import sys
import pandas as pd # pandas
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier ,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression ,SGDClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from time import time
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import randint as sp_randint
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from random import shuffle
from datetime import timedelta
import datetime
# import xgboost as xgb
# from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
# from lasagne.updates import nesterov_momentum,adagrad
# from lasagne.objectives import binary_crossentropy
# from nolearn.lasagne import NeuralNet
# import theano
# from theano import tensor as T
# from theano.tensor.nnet import sigmoid
from sklearn import metrics
from sklearn.utils import shuffle

########################################################################################################################
#West Nile Virus Prediction                                                                                            #
########################################################################################################################

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS,Weather_DS):

    print("***************Starting Data cleansing***************")
    ####################################################################################################################
    species_map = {'CULEX RESTUANS' : "100000",
                  'CULEX TERRITANS' : "010000",
                  'CULEX PIPIENS'   : "001000",
                  'CULEX PIPIENS/RESTUANS' : "101000",
                  'CULEX SALINARIUS': "000010",
                  'CULEX TARSALIS' :  "000001",
                  'UNSPECIFIED CULEX': "001000"} # Treating unspecified as PIPIENS (http://www.ajtmh.org/content/80/2/268.full)

    #Use all values in Species and create different column for Train and Actual DS
    for column in ['Species']:
        dummies = pd.get_dummies(Train_DS[column])
        Train_DS[dummies.columns] = dummies

    for column in ['Species']:
        dummies = pd.get_dummies(Actual_DS[column])
        Actual_DS[dummies.columns] = dummies

    Train_DS['CULEX PIPIENS']  = np.where (Train_DS['CULEX PIPIENS/RESTUANS'] == 1 , 1, Train_DS['CULEX PIPIENS'])
    Train_DS['CULEX RESTUANS'] = np.where (Train_DS['CULEX PIPIENS/RESTUANS'] == 1 , 1, Train_DS['CULEX RESTUANS'])
    Actual_DS['CULEX PIPIENS']  = np.where (Actual_DS['CULEX PIPIENS/RESTUANS'] == 1 , 1, Actual_DS['CULEX PIPIENS'])
    Actual_DS['CULEX RESTUANS'] = np.where (Actual_DS['CULEX PIPIENS/RESTUANS'] == 1 , 1, Actual_DS['CULEX RESTUANS'])
    Actual_DS['CULEX PIPIENS']  = np.where (Actual_DS['UNSPECIFIED CULEX'] == 1 , 1, Actual_DS['CULEX PIPIENS'])

    Train_DS = Train_DS.drop(['Species','CULEX PIPIENS/RESTUANS'], axis=1)
    Actual_DS = Actual_DS.drop(['Species','CULEX PIPIENS/RESTUANS','UNSPECIFIED CULEX'], axis=1)
    ####################################################################################################################


    #Not using code sum right now *********************************
    #Weather_DS = Weather_DS.drop('CodeSum', axis=1)

    #use imputer for missing values
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)
    cols = Weather_DS.columns[2:len(Weather_DS.columns)]
    cols2 =Weather_DS.columns[0:2]
    Weather_New = Weather_DS[cols]
    Weather_New = imp.fit_transform(Weather_DS[cols])
    Weather_New2 = pd.concat([Weather_DS[cols2],pd.DataFrame(Weather_New)],keys=None,axis=1)
    Weather_New2.columns = Weather_DS.columns
    Weather_DS = Weather_New2

    # Split station 1 and 2 and join horizontally
    weather_stn1 = Weather_DS[Weather_DS['Station']==1]
    weather_stn2 = Weather_DS[Weather_DS['Station']==2]
    weather_stn_d = Weather_DS['Date']
    weather_stn1 = weather_stn1.drop(['Station'], axis=1)
    weather_stn2 = weather_stn2.drop(['Station'], axis=1)
    #Weather_DS = weather_stn1.merge(weather_stn2, on='Date')
    df = pd.concat([weather_stn1, weather_stn2])
    Weather_DS = df.groupby(['Date']).mean().reset_index()


    ###################################################################################################################
    Weather_DS['Date']=pd.to_datetime(Weather_DS['Date'])
    Weather_DS_orig = Weather_DS.copy()

    for days_ago in [1,2,3,5,8,12]:
        Weather_DS_New = Weather_DS_orig.copy()
        Weather_DS_New['Date'] = pd.DatetimeIndex(Weather_DS_New['Date']) + pd.DateOffset(days_ago)
        Weather_DS = Weather_DS.merge(Weather_DS_New, on='Date')

    ###################################################################################################################

    #Weather_DS.to_csv(file_path+'Weather_DS_temp.csv')

    #Get month and Day from Date feature
    temp = pd.DatetimeIndex(Train_DS['Date'])
    Train_DS['month'] = temp.month
    Train_DS['day']   = temp.day
    Train_DS['year']   = temp.year
    Train_DS['week']   = temp.week

    temp = pd.DatetimeIndex(Actual_DS['Date'])
    Actual_DS['month'] = temp.month
    Actual_DS['day']   = temp.day
    Actual_DS['year']   = temp.year
    Actual_DS['week']   = temp.week

    # Add integer latitude/longitude columns
    # Train_DS['Lat_int'] = Train_DS.Latitude.apply(int)
    # Train_DS['Long_int'] = Train_DS.Longitude.apply(int)
    # Actual_DS['Lat_int'] = Actual_DS.Latitude.apply(int)
    # Actual_DS['Long_int'] = Actual_DS.Longitude.apply(int)



    # Convert categorical data to numbers
    lbl = preprocessing.LabelEncoder()
    # lbl.fit(list(Train_DS['Species'].values) + list(Actual_DS['Species'].values))
    # Train_DS['Species'] = lbl.transform(Train_DS['Species'].values)
    # Actual_DS['Species'] = lbl.transform(Actual_DS['Species'].values)

    lbl.fit(list(Train_DS['Street'].values) + list(Actual_DS['Street'].values))
    Train_DS['Street'] = lbl.transform(Train_DS['Street'].values)
    Actual_DS['Street'] = lbl.transform(Actual_DS['Street'].values)

    lbl.fit(list(Train_DS['Trap'].values) + list(Actual_DS['Trap'].values))
    Train_DS['Trap'] = lbl.transform(Train_DS['Trap'].values)
    Actual_DS['Trap'] = lbl.transform(Actual_DS['Trap'].values)

    Train_DS['Longitude'] = Train_DS['Longitude'] * -1
    Actual_DS['Longitude'] = Actual_DS['Longitude'] * -1

    # drop address columns
    Train_DS = Train_DS.drop(['Address', 'AddressNumberAndStreet'], axis = 1)
    Actual_DS = Actual_DS.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)

    #group all train recods
    Train_DS = Train_DS.groupby(['Date','Species','Block','Street','Trap','Latitude',
                     'Longitude','AddressAccuracy','month','day','year','week'])\
                        .agg({'NumMosquitos': np.sum , 'WnvPresent': np.max}).reset_index()

    #Insert NumMosquitos into Actual DS at same position as that of Train DS (not working in python 3.4)
    #Actual_DS.insert(Train_DS.columns.get_loc("NumMosquitos"), 'NumMosquitos', 1)
    Actual_DS['NumMosquitos'] = 1

    y = Train_DS.WnvPresent.values
    #Train_DS = Train_DS.drop(['WnvPresent'], axis = 1)

    Train_DS['Date']=pd.to_datetime(Train_DS['Date'])
    Actual_DS['Date']=pd.to_datetime(Actual_DS['Date'])

    # Merge with weather data
    Train_DS = Train_DS.merge(Weather_DS, on='Date')
    Actual_DS = Actual_DS.merge(Weather_DS, on='Date')

    Train_DS = Train_DS.drop(['Date'], axis = 1)
    Actual_DS = Actual_DS.drop(['Date'], axis = 1)

    Train_DS.to_csv(file_path+'Train_DS_New2.csv',index_label='id')
    Actual_DS.to_csv(file_path+'Actual_DS_New2.csv',index_label='id')

    print("***************Ending Data cleansing***************")

    return Train_DS, Actual_DS, y

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    global file_path
    file_path = 'C:/Python/Others/data/Kaggle/West_Nile_Virus_Prediction/'
    #file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/West_Nile_Virus_Prediction/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS =  pd.read_csv(file_path+'train.csv',sep=',')
    Actual_DS =  pd.read_csv(file_path+'test.csv',sep=',')
    Sample_DS = pd.read_csv(file_path+'sampleSubmission.csv',sep=',')
    Weather_DS = pd.read_csv(file_path+'weather4.csv',sep=',')

    Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS,Weather_DS)

########################################################################################################################
#Get the predictions for actual data set
########################################################################################################################


########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)

