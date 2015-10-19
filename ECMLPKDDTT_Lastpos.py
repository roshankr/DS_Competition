import json
import zipfile
import numpy as np
import pandas as pd
import sys
import platform

### Control the number of trips read for training 
### Control the number of closest trips used to calculate trip duration
### Parameters for train set trips to keep and how much to pad travel time based on polyline
N_read = 80000
N_trips = 100
P_keep = 0.95
P_pad = 1.25

### Get Haversine distance
def get_dist(lonlat1, lonlat2):
  lon_diff = np.abs(lonlat1[0]-lonlat2[0])*np.pi/360.0
  lat_diff = np.abs(lonlat1[1]-lonlat2[1])*np.pi/360.0
  a = np.sin(lat_diff)**2 + np.cos(lonlat1[1]*np.pi/180.0) * np.cos(lonlat2[1]*np.pi/180.0) * np.sin(lon_diff)**2  
  d = 2*6371*np.arctan2(np.sqrt(a), np.sqrt(1-a))
  return(d)

if(platform.system() == "Windows"):
    file_path = 'C:/Python/Others/data/Kaggle/ECMLPKDD_Taxi_Trajectory/'
else:
    file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/ECMLPKDD_Taxi_Trajectory/'


### Read test
#zf = zipfile.ZipFile('../input/test.csv.zip')
test = pd.read_csv(file_path+'test.csv',sep=',', usecols=['TRIP_ID', 'POLYLINE'])
print("reading test data")
test['POLYLINE'] = test['POLYLINE'].apply(json.loads)
test['snapshots'] = test['POLYLINE'].apply(len)
test['lonlat'] = test['POLYLINE'].apply(lambda x: x[0])
test.drop('POLYLINE', axis=1, inplace=True)

### Read train
#zf = zipfile.ZipFile('../input/train.csv.zip')
#train = pd.read_csv(zf.open('train.csv'), usecols=['POLYLINE'], nrows=N_read)

print("reading train data")
train = pd.read_csv(file_path+'train.csv',sep=',', usecols=['POLYLINE'])
print(np.shape(train))
train['POLYLINE'] = train['POLYLINE'].apply(json.loads)
train['snapshots'] = train['POLYLINE'].apply(len)
train = train[train.snapshots>25]
train['lonlat'] = train['POLYLINE'].apply(lambda x: x[0])
train.drop('POLYLINE', axis=1, inplace=True)

print("loop starts")
test['TRAVEL_TIME'] = 0
for row, ll in enumerate(test['lonlat']):	
   	### Weighted mean of trip duration
   	### Bound below by 10 meters since we use 1/distance^2 as weight
  	### Treat 5% of longest lasting trips as outliers  	
	d = train['lonlat'].apply(lambda x: get_dist(x, ll))
	i = np.argpartition(d, N_trips)[0:N_trips]
	w = np.maximum(d.iloc[i], 0.01)
	s = train.iloc[i]['snapshots']
	j = np.argpartition(s, int(N_trips*P_keep))[0:int(N_trips*P_keep)]
	test.loc[row, 'TRAVEL_TIME'] = np.maximum(P_pad*test.loc[row, 'snapshots'], np.average(s.iloc[j], weights=1/w.iloc[j]**2))
	
### Blend with test average
test['TRAVEL_TIME'] = 15*(.5*test['TRAVEL_TIME']+.5*P_pad*np.maximum(test.snapshots.mean(), test['snapshots']))
test['TRAVEL_TIME'] = test['TRAVEL_TIME'].astype(int)
test[['TRIP_ID', 'TRAVEL_TIME']].to_csv(file_path+'output/submission.csv', index=False)
