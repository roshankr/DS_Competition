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
from prettytable import PrettyTable
from prettytable import MSWORD_FRIENDLY

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

    #Train_DS.to_csv(file_path+'output_temp.csv')

    print("***************Ending Data cleansing***************")

    return Train_DS

########################################################################################################################
#Get the best neighbor
########################################################################################################################
def Get_Best_Neigbhbor(Train_DS, i , k, n_cnt):

    # Best neighbour
    best_gain = 0
    best_neigbhor = 0
    curr_gain = 0

    for j in range(n_cnt):

        Dist_col = 'Dist_'+str(j+k+1)
        Weight_col = 'Weight_'+str(j+k+1)
        DD_col = 'DD_'+str(j+k+1)
        NN_col = 'NN_'+str(j+k+1)
        next_i = (Train_DS[Train_DS['GiftId'] == Train_DS.ix[i, NN_col]].index)[0]

        if cost_function == 1:
            ##------------------------------------------------------------------------------------------------------------##
            #gain formula - Combination of Weight and Distance
            curr_gain =  (Train_DS.ix[i, Weight_col]/ Train_DS.ix[i, Dist_col])  + (10/(abs(Train_DS.ix[i, DD_col])))
            ##------------------------------------------------------------------------------------------------------------##
        elif cost_function == 2:
             #gain formula - Combination of Weight and Distance
            #curr_gain =  (Train_DS.ix[i, Weight_col]/ Train_DS.ix[i, Dist_col])  + (10/(abs(Train_DS.ix[i, DD_col])))
            curr_gain =  (Train_DS.ix[i, Weight_col]/ Train_DS.ix[i, Dist_col])
            ##------------------------------------------------------------------------------------------------------------##
        elif cost_function == 3:
            # #Take neighbor with least Weight
            curr_gain =  1 / Train_DS.ix[i, Weight_col]
            ##------------------------------------------------------------------------------------------------------------##
        elif cost_function == 4:
            # #Take neighbor with max Weight
            curr_gain =  Train_DS.ix[i, Weight_col]
            ##------------------------------------------------------------------------------------------------------------##
        elif cost_function == 5:
            # #Take neighbor with least Distance
            curr_gain =  1 / Train_DS.ix[i, Dist_col]
            ##------------------------------------------------------------------------------------------------------------##
        elif cost_function == 6:
            # #Take neighbor with max Distance
            curr_gain =  Train_DS.ix[i, Dist_col]
            ##------------------------------------------------------------------------------------------------------------##
        #consider this one only if it has the best gain and its not yet considered (not assigned a trip count)
        if curr_gain > best_gain and Train_DS.ix[next_i, 'TripId']==0:
            best_gain = curr_gain
            best_neigbhor = Train_DS.ix[i, NN_col]

    return best_gain, best_neigbhor

########################################################################################################################
#Get the next best node by traversing back
########################################################################################################################
def Get_next_best_node_Vertical(i , Train_DS, n_cnt):

    # Best neighbour
    i = Train_DS.ix[i, 'predecessor'].astype(int)

    #curr_trip = Train_DS.ix[i, 'TripId']
    j = 0
    not_found_gain = True
    predecessors = 0

    while (not_found_gain):
        predecessors = predecessors + 1
        best_gain, best_neigbhor = Get_Best_Neigbhbor(Train_DS, i , j, No_Horizontal_nodes)

        if best_gain > 0:
            not_found_gain = False
            i = (Train_DS[Train_DS['GiftId'] == best_neigbhor].index)[0]
        else:
            i = Train_DS.ix[i, 'predecessor'].astype(int)

        if predecessors >= No_Neighbor_nodes:
            i = 0
            not_found_gain = False

    #Train_DS.to_csv(file_path+'Train_with_NN2.csv')

    return i

########################################################################################################################
#Get the next best node by traversing back
########################################################################################################################
def Get_next_best_node_Horizontal(i , Train_DS, n_cnt):

    # Best neighbour
    cnt = 0
    not_found_gain = True
    j = 0

    #Find the neighbors Weights
    while (not_found_gain):

        cnt = cnt + 1
        prev = Train_DS.ix[i,'NN_'+str(cnt)].astype(int)
        new_i =(Train_DS[Train_DS['GiftId'] == prev].index)[0]

        best_gain, best_neigbhor = Get_Best_Neigbhbor(Train_DS, new_i , j, No_Horizontal_nodes)

        if best_gain > 0:
            not_found_gain = False
            i = (Train_DS[Train_DS['GiftId'] == best_neigbhor].index)[0]

        if cnt >= No_Neighbor_nodes:
            not_found_gain = False
            i = 0
            
    return i

########################################################################################################################
#Identify the groups
########################################################################################################################
def Identify_groups(Train_DS, n_cnt):
    print("***************Starting Identify the groups***************")
    print("Manual Iteration....... at Time: %s" %(tm.strftime("%H:%M:%S")))

    Train_DS['TripId'] = 0
    Train_DS['TOrder'] = 0
    TOrder = 1
    TripId = 1
    Tot_Weight = 0
    i = 0
    prev_i = 0
    going_back_node = 0
    First_val = 0
    Last_val = 0
    check_all_trip = False
    loop_cnt = 0

    while not(check_all_trip):

        if loop_cnt % 1000 == 0:
            print("%d - Iteration..........at Time: %s" %(loop_cnt,tm.strftime("%H:%M:%S")))

        if loop_cnt == 0:
            First_val = Train_DS.ix[i, 'GiftId']

        ##------------------------------------------------------------------------------------------------------------##
        #reset trip if weight crosses 1000

        if (Tot_Weight + Train_DS.ix[i, 'Weight'] > 1000):
            TripId = TripId + 1
            Tot_Weight = 0
            TOrder = 1
            i = (Train_DS[Train_DS['TripId']==0].head(1)).index[0]
            First_val = Train_DS.ix[i, 'GiftId']

        Train_DS.ix[i, 'predecessor']    =  prev_i
        ##------------------------------------------------------------------------------------------------------------##

        not_found_gain = True
        k = 0
        while (not_found_gain):
            best_gain, best_neigbhor = Get_Best_Neigbhbor(Train_DS, i , k, 5)
            k = k + 5

            if best_gain > 0 or k > (n_cnt-5):
               not_found_gain = False

        ##------------------------------------------------------------------------------------------------------------##
        Train_DS.ix[i, 'TripId']    =  TripId
        Tot_Weight = Tot_Weight + Train_DS.ix[i, 'Weight']
        Train_DS.ix[i, 'Tot_Weight']    =  Tot_Weight
        Train_DS.ix[i, 'comment']    =  'G'
        Train_DS.ix[i, 'First_val'] = First_val
        Train_DS.ix[i, 'TOrder'] = TOrder
        TOrder = TOrder + 1

        if best_gain > 0:
            Train_DS.ix[i, 'best_gain']     =  best_gain
            Train_DS.ix[i, 'best_neigbhor'] =  best_neigbhor
            prev_i = i
            i = (Train_DS[Train_DS['GiftId'] == best_neigbhor].index)[0]

        else:
            Train_DS.ix[i, 'best_gain']     =  0
            Train_DS.ix[i, 'best_neigbhor'] =  0
            prev_i = i

            if not(Train_DS[Train_DS['TripId']==0].empty):
                #i = (Train_DS[Train_DS['TripId']==0].head(1)).index[0]

                #check my  nighbor's neighbor
                new_i = Get_next_best_node_Horizontal(i , Train_DS, n_cnt)
                if new_i == 0:
                    #check my predecssor's neighbor
                    new_i = Get_next_best_node_Vertical(i , Train_DS, n_cnt)
                    if new_i == 0:
                        # if no one gets ,the get the next unmarked one
                        #i = (Train_DS[Train_DS['TripId']==0].head(1)).index[0]
                        Tot_Weight = 1001

                    else:
                        i = new_i
                else:
                    i = new_i

            else:
                check_all_trip = True

            First_val = Train_DS.ix[i, 'GiftId']

        loop_cnt = loop_cnt + 1

    #Train_DS.to_csv(file_path+'Train_with_NN2.csv')

    print("***************Ending Identify the groups***************")

    return Train_DS

########################################################################################################################
#Get the final cost optimization
########################################################################################################################
def Get_Cost_Optimization(Train_DS):

    #Associate Cost for each unit and Trip

    Train_DS['tot_cost'] = 0
    cols_valid = ['GiftId','Latitude','Longitude','Weight','TripId']

    last_TripId = Train_DS.iloc[-1]['TripId']

    for trip in range(Train_DS.TripId.nunique()):

        trip = trip + 1
        temp = Train_DS[Train_DS['TripId']==trip]
        val_err = weighted_reindeer_weariness(temp[cols_valid])
        Train_DS[Train_DS['TripId']==trip]['tot_cost'] = val_err

        if val_err > 12000000:
            Break_TripId_Optimization(Train_DS,trip, last_TripId )

    sys.exit(0)

########################################################################################################################
#Break_TripId_Optimization
########################################################################################################################
def Break_TripId_Optimization(Train_DS,trip, last_TripId):

    temp = Train_DS[Train_DS['TripId']==trip]
    temp['new_TripId'] = 0
    temp['cost'] = 0
    tot_cost = 0
    prev_cost = 0
    single_cost = 0

    cols_valid = ['GiftId','Latitude','Longitude','Weight','TripId']

    row_iterator = temp.iterrows()
    for i, row in row_iterator:
        temp.ix[i,'new_TripId'] = 1

        single_cost = weighted_reindeer_weariness(temp.ix[cols_valid])
        tot_cost = tot_cost + weighted_reindeer_weariness(temp[cols_valid])

        if (tot_cost - prev_cost) >

        temp.ix[i,'cost'] = tot_cost

        prev_cost = tot_cost

    
########################################################################################################################
#Kmeans_Clustering
########################################################################################################################
def Manual_Clustering(Train_DS):

    print("***************Starting Manual Clustering***************")

    ##----------------------------------------------------------------------------------------------------------------##
    print("Starting NearestNeighbors Iteration....... at Time: %s" %(tm.strftime("%H:%M:%S")))

    cols = ['GiftId','Latitude','Longitude']
    Temp_DS = Train_DS[cols].set_index('GiftId')
    Temp_DS['Latitude']  = Temp_DS['Latitude'].apply(lambda X: radians(X))
    Temp_DS['Longitude'] = Temp_DS['Longitude'].apply(lambda X: radians(X))

    ##----------------------------------------------------------------------------------------------------------------##

    clf = NearestNeighbors(n_neighbors=neighbor_cnt+1,metric='haversine')
    clf.fit(Temp_DS)
    dis , neighbors = clf.kneighbors(Temp_DS,neighbor_cnt+1,return_distance=True)
    dis = dis * 6371 #Average Earth Radius

    #Delete the first column as it point to the same data point
    dis = sp.delete(dis, 0, 1)
    neighbors = sp.delete(neighbors, 0, 1)

    Dist_col = []
    NN_col = []
    for cnt in range(neighbor_cnt):
        cnt = cnt + 1

        Dist_col.append('Dist_'+str(cnt))
        NN_col.append('NN_'+str(cnt))

    dis = pd.DataFrame(dis,columns=Dist_col)
    neighbors = pd.DataFrame(neighbors,columns=NN_col)
    neighbors = neighbors + 1

    # Concat Distiance to the neighbors
    Train_DS = pd.concat([Train_DS, dis],axis=1)
    # Concat neighbors
    Train_DS = pd.concat([Train_DS, neighbors],axis=1)

    #Find the neighbors Weights
    for cnt in range(neighbor_cnt):
        cnt = cnt + 1
        Train_DS['Weight_'+str(cnt)] = np.array(Train_DS.ix[Train_DS['NN_'+str(cnt)],'Weight'].reset_index(drop=True))

    #Find the neighbors Distances from source
    for cnt in range(neighbor_cnt):
        cnt = cnt + 1
        Train_DS['DD_'+str(cnt)] = np.array(Train_DS.ix[Train_DS['NN_'+str(cnt)],'DD'].reset_index(drop=True))

    Train_DS = Train_DS.sort(columns='Distance',ascending=True).reset_index(drop=True)

    #Train_DS.to_csv(file_path+'Train_with_NN.csv')

    print("Ending NearestNeighbors Iteration....... at Time: %s" %(tm.strftime("%H:%M:%S")))

    ##----------------------------------------------------------------------------------------------------------------##

    Train_DS = Identify_groups(Train_DS,neighbor_cnt)

    Train_DS = Train_DS.sort(columns=['TripId','TOrder'], ascending = ['True','True'])
    #Train_DS.to_csv(file_path+'Train_with_NN2.csv')

    Train_DS = Get_Cost_Optimization(Train_DS)

    cols_valid = ['GiftId','Latitude','Longitude','Weight','TripId']

    print("<-------------------------------Summary------------------------------------>")
    total = 0

    Weight_sum = list(Train_DS.groupby('TripId').Weight.sum())
    Weight_cnt = list(Train_DS.groupby('TripId').Weight.count())

    pt = PrettyTable(['TripId', 'Count', 'Weight', 'Cost'])
    pt.align['TripId'] = "r"
    pt.align['Count'] = "r"
    pt.align['Weight'] = "r"
    pt.align['Cost'] = "r"

    for trip in range(Train_DS.TripId.nunique()):
        trip = trip + 1
        temp = Train_DS[Train_DS['TripId']==trip]
        val_err = weighted_reindeer_weariness(temp[cols_valid])
        total = total + val_err
        val_err = '${:,.2f}'.format(val_err)

        wt_sum = '{:,.2f}'.format(Weight_sum[trip-1])
        pt.add_row([trip, Weight_cnt[trip-1],wt_sum,val_err ])

    total = '${:,.2f}'.format(total)
    pt.add_row(['-----------', '-----------','-----------','---------------------' ])
    pt.add_row(['Total('+str(len(Train_DS))+'-'+str(neighbor_cnt)+'-'+str(cost_function)+')',sum(Weight_cnt),sum(Weight_sum), total ])
    pt.add_row(['-----------', '-----------','-----------','---------------------' ])
    pt.add_row(['Actual('+str(len(Train_DS))+'-'+str(neighbor_cnt)+'-'+str(cost_function)+')',len(Train_DS),Train_DS.Weight.sum(), total ])
    print(pt)

    print("<------------------------------------------------------------------------->")

    cols_valid = ['GiftId','TripId']
    Train_DS = Train_DS[cols_valid]
    Train_DS.to_csv(file_path+'output/Submission_Roshan_NN_1.csv')

    print("***************Ending Manual Clustering***************")
    return Train_DS

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, Featimp_DS,  hav_dist, header, cost_function, neighbor_cnt, No_Horizontal_nodes,No_Neighbor_nodes

    hav_dist = metrics.make_scorer(haversine_dist)

    random.seed(21)
    np.random.seed(21)

    #Parms
    #CF - 1 --> With Weight and Distance Combination
    #CF - 2 --> With Weight and Distance Combination (no return)
    #CF - 3 --> Take neighbor with least Weight
    #CF - 4 --> Take neighbor with max Weight
    #CF - 5 --> Take neighbor with least Distance ***best
    #CF - 6 --> Take neighbor with max Distance

    cost_function = 5
    neighbor_cnt = 20
    No_Horizontal_nodes = 20
    No_Neighbor_nodes = 20
    header = 1000

    if(platform.system() == "Windows"):

        file_path = 'C:/Python/Others/data/Kaggle/Santa_Stolen_Sleigh/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Santa_Stolen_Sleigh/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS      = pd.read_csv(file_path+'gifts.csv',sep=',')

    Train_DS = Train_DS.head(header)

    Train_DS =  Data_Munging(Train_DS)

    pred_Actual  = Manual_Clustering(Train_DS)
########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)


########################################################################################################################
#Comments                                                                                                              #
########################################################################################################################
'''
with :
    cost_function = 1,2,3,4,5,6
    neighbor_cnt = 20
    No_Horizontal_nodes = 20
    No_Neighbor_nodes = 20

#  Total(2000-20-1) |        2000 | 27001.9539944 |       $568,903,845.46 |
#  Total(2000-20-2) |        2000 | 27001.9539944 |       $536,260,386.97 |
#  Total(2000-20-3) |        2000 | 27001.9539944 |       $563,331,953.39 |
#  Total(2000-20-4) |        2000 | 27001.9539944 |       $527,636,804.47 |
#  Total(2000-20-5) |        2000 | 27001.9539944 |       $484,873,169.41 |----
#  Total(2000-20-6) |        2000 | 27001.9539944 |       $631,534,727.55 |


with :
    cost_function = 5
    neighbor_cnt = 20
    No_Horizontal_nodes = 20
    No_Neighbor_nodes = 20


|  Total(2000-20-5)   |        2000 | 27001.9539944 |       $481,723,362.37 |%
|  Total(5000-20-5)   |        5000 | 68583.7634855 |     $1,046,972,813.12 |
|  Total(10000-20-5)  |       10000 | 137493.009662 |     $1,940,780,426.43 |194078.042643
|  Total(25000-20-5)  |       25000 | 348646.632396 |     $4,277,395,417.37 |171095.816694
|  Total(25000-30-5)  |       25000 | 348646.632396 |     $4,216,284,387.50 |
|  Total(25000-40-5)  |       25000 | 348646.632396 |     $4,173,259,722.21 |
|  Total(25000-50-5)  |       25000 | 348646.632396 |     $4,210,948,398.77
|  Total(50000-20-5)  |       50000 | 703758.113225 |     $8,153,157,648.77 |163063.152975
|  Total(75000-20-5)  |       75000 | 1057511.71831 |    $12,106,329,812.83 |161417.730837
|  Total(100000-20-5) |      100000 | 1409839.09802 |    $15,723,475,409.85 |157234.7540985
							                                         Target : 123063.2981283

'''