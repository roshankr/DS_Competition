import requests
import numpy as np
import scipy as sp
import sys
import platform
import pandas as pd
from time import time
from operator import itemgetter
import re
import time as tm
from math import radians, cos, sin, asin, sqrt
import random
import warnings
from math import sqrt, exp, log
from datetime import date,timedelta as td,datetime as dt
import datetime
from sklearn.cluster import *
from sklearn.neighbors import *
from sklearn import metrics

########################################################################################################################
#Santa's Stolen Sleigh
########################################################################################################################
#--------------------------------------------Algorithm : Random Forest :------------------------------------------------
#Random Forest :
#--------------------------------------------Algorithm : XGB------------------------------------------------------------
#XGB :

#--------------------------------------------Suggestions, Ideas---------------------------------------------------------
#Suggestions, Ideas
########################################################################################################################

def weighted_trip_length(stops, weights):
    north_pole = (90,0)
    sleigh_weight = 10

    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)

    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)
    for location, weight in zip(tuples, weights):
        dist = dist + haversine_dist(location, prev_stop) * prev_weight
        prev_stop = location
        prev_weight = prev_weight - weight
    return dist

########################################################################################################################
def weighted_reindeer_weariness(all_trips):
    weight_limit = 1000
    uniq_trips = all_trips.TripId.unique()

    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")

    dist = 0.0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId==t]
        dist = dist + weighted_trip_length(this_trip[['Latitude','Longitude']], this_trip.Weight.tolist())

    return dist

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid(X):

    print("***************Starting Cross validation***************")

    clf = KMeans(n_clusters=100, max_iter=10 )

    cols = ['GiftId','DD']
    clf.fit(X[cols])
    pred_Actual = clf.predict(X[cols])

    X['TripId'] = pred_Actual

    cols_valid = ['GiftId','Latitude','Longitude','Weight','TripId']


    #val_err = weighted_reindeer_weariness(X[cols_valid])

    print(X.head())

    print(val_err)

    sys.exit(0)

    print("***************Ending Kfold Cross validation***************")

    return pred_Actual

########################################################################################################################
# Calculate the great-circle distance bewteen two points on the Earth surface.
# :input: two 2-tuples, containing the latitude and longitude of each point in decimal degrees.
# Example: haversine((45.7597, 4.8422), (48.8567, 2.3508)) ,:output: Returns the distance bewteen the two points.
# The default unit is kilometers. Miles can be returned if the ``miles`` parameter is set to True.
########################################################################################################################
def haversine_dist(point1, point2, miles=False):

    AVG_EARTH_RADIUS = 6371  # in km

    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lng / 2) ** 2
    h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers

def haversine_dist1(point1, point2, miles=False):

    print(point1)
    print(point2)
    sys.exit(0)
    AVG_EARTH_RADIUS = 6371  # in km

    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lng / 2) ** 2
    h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers

########################################################################################################################
#Get Haverine Distance
########################################################################################################################
def Hav_Dist(row):

    point1 = (90,0)
    point2 = (row['Latitude'],row['Longitude'])
    dist = haversine_dist(point1, point2)

    return dist

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS):

    print("***************Starting Data cleansing***************")

    global  Train_DS1

    print("Getting Distance between the points in Km")
    Train_DS['Distance'] = Train_DS.apply(Hav_Dist, axis = 1).astype(float)

    print("Getting Direction, Towards West = -1 , East = 1 ")
    Train_DS['Direction'] = np.where(Train_DS['Longitude'] <=0 , '-1','1').astype(int)

    Train_DS['DD'] = Train_DS['Distance'] * Train_DS['Direction']

    Train_DS = Train_DS.drop(['Direction'], axis = 1).reset_index(drop=True)

    Train_DS.to_csv(file_path+'output_temp.csv')

    print("***************Ending Data cleansing***************")

    return Train_DS

########################################################################################################################
#Kmeans_Clustering
########################################################################################################################
def Kmeans_Clustering(Train_DS, Sample_DS):
    print("***************Starting Clustering***************")
    t0 = time()

    pred_Actual = Nfold_Cross_Valid(Train_DS)

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.GiftId.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_Kmeans_1.csv', index_label='GiftId')

    print("***************Ending Clustering***************")
    return pred_Actual



########################################################################################################################
#Kmeans_Clustering
########################################################################################################################
def Manual_Clustering(Train_DS, Sample_DS):

    print("***************Starting Manual Clustering***************")

    ##----------------------------------------------------------------------------------------------------------------##
    print("Starting NearestNeighbors Iteration....... at Time: %s" %(tm.strftime("%H:%M:%S")))
    cols = ['Latitude','Longitude']
    Temp_DS = Train_DS[cols]
    Temp_DS['Latitude']  = Temp_DS['Latitude'].apply(lambda X: radians(X))
    Temp_DS['Longitude'] = Temp_DS['Longitude'].apply(lambda X: radians(X))

    clf = NearestNeighbors(n_neighbors=6,metric='haversine')
    clf.fit(Temp_DS)
    dis , neighbors = clf.kneighbors(Temp_DS,6,return_distance=True)
    dis = dis * 6371 #Average Earth Radius

    #Delete the first column as it point to the same data point
    dis = sp.delete(dis, 0, 1)
    neighbors = sp.delete(neighbors, 0, 1)

    dis = pd.DataFrame(dis,columns=['Dist_1','Dist_2','Dist_3','Dist_4','Dist_5'])
    neighbors = pd.DataFrame(neighbors,columns=['NN_1','NN_2','NN_3','NN_4','NN_5'])

    Train_DS = pd.concat([Train_DS, dis],axis=1)
    Train_DS = pd.concat([Train_DS, neighbors],axis=1)

    Train_DS['Weight_1'] = np.array(Train_DS.ix[Train_DS['NN_1'],'Weight'].reset_index(drop=True))
    Train_DS['Weight_2'] = np.array(Train_DS.ix[Train_DS['NN_2'],'Weight'].reset_index(drop=True))
    Train_DS['Weight_3'] = np.array(Train_DS.ix[Train_DS['NN_3'],'Weight'].reset_index(drop=True))
    Train_DS['Weight_4'] = np.array(Train_DS.ix[Train_DS['NN_4'],'Weight'].reset_index(drop=True))
    Train_DS['Weight_5'] = np.array(Train_DS.ix[Train_DS['NN_5'],'Weight'].reset_index(drop=True))

    Train_DS['DD_1'] = np.array(Train_DS.ix[Train_DS['NN_1'],'DD'].reset_index(drop=True))
    Train_DS['DD_2'] = np.array(Train_DS.ix[Train_DS['NN_2'],'DD'].reset_index(drop=True))
    Train_DS['DD_3'] = np.array(Train_DS.ix[Train_DS['NN_3'],'DD'].reset_index(drop=True))
    Train_DS['DD_4'] = np.array(Train_DS.ix[Train_DS['NN_4'],'DD'].reset_index(drop=True))
    Train_DS['DD_5'] = np.array(Train_DS.ix[Train_DS['NN_5'],'DD'].reset_index(drop=True))

    Train_DS = Train_DS.sort(columns='Distance',ascending=True)

    #Train_DS.to_csv(file_path+'Train_with_NN.csv')

    print("Ending NearestNeighbors Iteration....... at Time: %s" %(tm.strftime("%H:%M:%S")))

    ##----------------------------------------------------------------------------------------------------------------##
    print("Kmeans Iteration....... at Time: %s" %(tm.strftime("%H:%M:%S")))
    cols = ['Latitude','Longitude']
    clf = KMeans(n_clusters=100, max_iter=10 )
    clf.fit(Train_DS[cols])
    pred_Actual = clf.predict(Train_DS[cols])
    Train_DS['clusters'] = pred_Actual
    ##----------------------------------------------------------------------------------------------------------------##


    #sort with DD (distance from origin) as a starting point
    Train_DS = Train_DS.sort(columns=['clusters','Distance']).reset_index(drop=True)
    Train_DS =  Train_DS.head(5000)
    print(Train_DS.head())

    sum_weight = 0
    Train_DS['TripId'] = 0
    Train_DS['sum_wt'] = 0
    trip_id = 1
    prev_clust = 0

    print("Manual Iteration....... at Time: %s" %(tm.strftime("%H:%M:%S")))
    for i, row in Train_DS.iterrows():

        if i % 1000 == 0:
            print("%d - Iteration..........at Time: %s" %(i,tm.strftime("%H:%M:%S")))

        if (sum_weight + Train_DS.ix[i, 'Weight']) <= 1000:
            sum_weight = sum_weight + Train_DS.ix[i, 'Weight']
            Train_DS.ix[i, 'TripId'] = trip_id
            Train_DS.ix[i, 'sum_wt'] = sum_weight

        else:
            sum_weight = 0
            trip_id = trip_id + 1
            sum_weight = sum_weight + Train_DS.ix[i, 'Weight']
            Train_DS.ix[i, 'TripId'] = trip_id
            Train_DS.ix[i, 'sum_wt'] = sum_weight

    Train_DS.to_csv(file_path+'output_temp2.csv')
    cols_valid = ['GiftId','Latitude','Longitude','Weight','TripId']
    val_err = weighted_reindeer_weariness(Train_DS[cols_valid])

    print(val_err)
    sys.exit(0)

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.GiftId.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_Kmeans_1.csv', index_label='GiftId')

    print("***************Ending Manual Clustering***************")
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, Train_DS1, Featimp_DS,  hav_dist

    hav_dist = metrics.make_scorer(haversine_dist)

    random.seed(21)
    np.random.seed(21)

    if(platform.system() == "Windows"):

        file_path = 'C:/Python/Others/data/Kaggle/Santa_Stolen_Sleigh/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Santa_Stolen_Sleigh/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS      = pd.read_csv(file_path+'gifts.csv',sep=',')
    Sample_DS     = pd.read_csv(file_path+'sample_submission.csv',sep=',')

    Train_DS =  Data_Munging(Train_DS)

    pred_Actual  = Manual_Clustering(Train_DS, Sample_DS)
    #pred_Actual  = Kmeans_Clustering(Train_DS, Sample_DS)
########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)