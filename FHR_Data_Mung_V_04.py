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
from sklearn.svm import SVC
from time import time
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import randint as sp_randint
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from random import shuffle
import datetime

########################################################################################################################
#Facebook - Human or Robot                                                                                             #
########################################################################################################################

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS, Actual_DS, Bids_DS, Sample_DS):

    print("***************Starting Data cleansing***************")

    Sample_DS['id'] = Sample_DS['bidder_id']

    Bids_DS['win'] = 0
    Bids_DS['time_diff'] = 0
    Bids_DS['consecutive_bid'] = 0
    Bids_DS['first_bid'] = 0
    Bids_DS['cost_of_auction'] = 0
    Bids_DS['win_cost_of_auction'] = 0

    Temp = Bids_DS.groupby(['auction']).size().reset_index()
    Temp.columns = ['auction', 'no_of_bids_auction']
    Bids_DS = Bids_DS.merge(Temp, on='auction')

    Bids_DS['date']     = Bids_DS['time'].apply(str).apply(lambda x: x[0:7])
    Bids_DS['time_val'] = Bids_DS['time'].apply(str).apply(lambda x: x[7:16])

    # print(pd.DataFrame(Bids_DS.time_val.unique()))
    # sys.exit(0)

    # Convert categorical data to numbers
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(Train_DS['payment_account'].values) + list(Actual_DS['payment_account'].values))
    Train_DS['payment_account'] = lbl.transform(Train_DS['payment_account'].values)
    Actual_DS['payment_account'] = lbl.transform(Actual_DS['payment_account'].values)

    lbl.fit(list(Train_DS['address'].values) + list(Actual_DS['address'].values))
    Train_DS['address'] = lbl.transform(Train_DS['address'].values)
    Actual_DS['address'] = lbl.transform(Actual_DS['address'].values)

    Bids_DS['auction'] = lbl.fit_transform(list(Bids_DS['auction'].values))
    Bids_DS['merchandise'] = lbl.fit_transform(list(Bids_DS['merchandise'].values))
    Bids_DS['device'] = lbl.fit_transform(list(Bids_DS['device'].values))
    Bids_DS['country'] = lbl.fit_transform(list(Bids_DS['country'].values))
    Bids_DS['ip'] = lbl.fit_transform(list(Bids_DS['ip'].values))
    Bids_DS['url'] = lbl.fit_transform(list(Bids_DS['url'].values))

    lbl.fit(list(Train_DS['bidder_id'].values) + list(Actual_DS['bidder_id'].values)+ list(Bids_DS['bidder_id'].values))
    Train_DS['bidder_id'] = lbl.transform(Train_DS['bidder_id'].values)
    Actual_DS['bidder_id'] = lbl.transform(Actual_DS['bidder_id'].values)
    Bids_DS['bidder_id'] = lbl.transform(Bids_DS['bidder_id'].values)
    Sample_DS['bidder_id'] = lbl.transform(Sample_DS['bidder_id'].values)

    #for each ip address check how many shared users
    Bids_DS_ip_check = Bids_DS.groupby(['ip'])

    Temp = pd.DataFrame(Bids_DS_ip_check.bidder_id.nunique()).reset_index()
    Temp.columns = ['ip', 'bidder_id_unique']
    Bids_DS = Bids_DS.merge(Temp, on='ip')
    Bids_DS['bidder_id_unique'] = 1/ Bids_DS['bidder_id_unique']

    Temp = pd.DataFrame(Bids_DS_ip_check.country.nunique()).reset_index()
    Temp.columns = ['ip', 'country_ip_unique']
    Bids_DS = Bids_DS.merge(Temp, on='ip')
    Bids_DS['country_ip_unique'] = 1/ Bids_DS['country_ip_unique']

    #Bids_DS.to_csv(file_path+'Bids_DS_Check.csv')
    #Bids_test = Bids_DS[Bids_DS['bidder_id'] == 3186]
    # print(Bids_test)

    Bids_Summary = pd.DataFrame()
    Bids_Summary['bidder_id'] = Bids_DS['bidder_id'].unique()

    Bids_DS_auction = Bids_DS.groupby(['bidder_id'])

    Temp = pd.DataFrame(Bids_DS_auction.bid_id.count()).reset_index()
    Bids_Summary = Bids_Summary.merge(Temp, on='bidder_id')

    #each Bidder - unique auctions, merchandise , ip , url etc.......
    Temp = pd.DataFrame(Bids_DS_auction.auction.nunique()).reset_index()
    Bids_Summary = Bids_Summary.merge(Temp, on='bidder_id')

    Bids_Summary['avg_bid'] = Bids_Summary['bid_id']  / Bids_Summary['auction']

    # no need to use merchandise, instead use separate column for each merchandise
    Temp = pd.DataFrame(Bids_DS_auction.merchandise.nunique()).reset_index()
    Bids_Summary = Bids_Summary.merge(Temp, on='bidder_id')

    Temp = pd.DataFrame(Bids_DS_auction.device.nunique()).reset_index()
    Bids_Summary = Bids_Summary.merge(Temp, on='bidder_id')

    Bids_Summary['bid_device'] = Bids_Summary['device']  / Bids_Summary['bid_id']

    Temp = pd.DataFrame(Bids_DS_auction.country.nunique()).reset_index()
    Bids_Summary = Bids_Summary.merge(Temp, on='bidder_id')

    Bids_Summary['bid_country'] = Bids_Summary['country']  / Bids_Summary['bid_id']

    Temp = pd.DataFrame(Bids_DS_auction.ip.nunique()).reset_index()
    Bids_Summary = Bids_Summary.merge(Temp, on='bidder_id')

    Bids_Summary['bid_ip'] = Bids_Summary['ip']  / Bids_Summary['bid_id']

    Temp = pd.DataFrame(Bids_DS_auction.bidder_id_unique.sum()).reset_index()
    Bids_Summary = Bids_Summary.merge(Temp, on='bidder_id')
    Bids_Summary['bidder_id_unique'] = Bids_Summary['bidder_id_unique']  / Bids_Summary['bid_id']

    Temp = pd.DataFrame(Bids_DS_auction.country_ip_unique.sum()).reset_index()
    Bids_Summary = Bids_Summary.merge(Temp, on='bidder_id')
    Bids_Summary['country_ip_unique'] = Bids_Summary['country_ip_unique']  / Bids_Summary['bid_id']

    Temp = pd.DataFrame(Bids_DS_auction.url.nunique()).reset_index()
    Bids_Summary = Bids_Summary.merge(Temp, on='bidder_id')

    #for each Bidder -  auctions, merchandise , ip , url etc.......
    Bids_DS_auction_next = Bids_DS.groupby(['bidder_id','auction'])

    # Temp = pd.DataFrame(Bids_DS_auction_next.no_of_bids_auction.count()).reset_index()
    # Temp.columns = ['bidder_id', 'auction_1', 'auc_bid_id']
    # print(Temp)
    # sys.exit(0)

    Temp = pd.DataFrame(Bids_DS_auction_next.device.nunique()).reset_index()
    Temp.columns = ['bidder_id', 'auction_1', 'auc_dev']
    Bids_DS_auction = Temp.groupby(['bidder_id']).agg({'auc_dev': [np.mean]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','auc_dev']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')
    print("done")

    Temp = pd.DataFrame(Bids_DS_auction_next.country.nunique()).reset_index()
    Temp.columns = ['bidder_id', 'auction_2', 'auc_country']
    Bids_DS_auction = Temp.groupby(['bidder_id']).agg({'auc_country': [np.mean]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','auc_country']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')
    print("done")

    Temp = pd.DataFrame(Bids_DS_auction_next.ip.nunique()).reset_index()
    Temp.columns = ['bidder_id', 'auction_3', 'auc_ip']
    Bids_DS_auction = Temp.groupby(['bidder_id']).agg({'auc_ip': [np.mean]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','auc_ip']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')
    print("done")

    Temp = pd.DataFrame(Bids_DS_auction_next.url.nunique()).reset_index()
    Temp.columns = ['bidder_id', 'auction_4', 'auc_url']
    Bids_DS_auction = Temp.groupby(['bidder_id']).agg({'auc_url': [np.mean]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','auc_url']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')
    print("done")

    #for each Bidder-auction-device-country, different ip
    Bids_DS_auction_next = Bids_DS.groupby(['bidder_id','auction','device'])

    Temp = pd.DataFrame(Bids_DS_auction_next.ip.nunique()).reset_index()
    Temp.columns = ['bidder_id', 'auction','device','auc_dev_ip_diff']
    Bids_DS_auction = Temp.groupby(['bidder_id']).agg({'auc_dev_ip_diff': [np.mean]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','auc_dev_ip_diff']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')

    Temp = pd.DataFrame(Bids_DS_auction_next.url.nunique()).reset_index()
    Temp.columns = ['bidder_id', 'auction','device','auc_dev_url_diff']
    Bids_DS_auction = Temp.groupby(['bidder_id']).agg({'auc_dev_url_diff': [np.mean]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','auc_dev_url_diff']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')

    Temp = pd.DataFrame(Bids_DS_auction_next.country.nunique()).reset_index()
    Temp.columns = ['bidder_id', 'auction','device','auc_dev_country_diff']
    Bids_DS_auction = Temp.groupby(['bidder_id']).agg({'auc_dev_country_diff': [np.mean]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','auc_dev_country_diff']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')

    Bids_Summary['ip_url_ratio'] = Bids_Summary['auc_dev_ip_diff'] / Bids_Summary['auc_dev_url_diff']
    Bids_Summary['ip_country_ratio'] = Bids_Summary['auc_dev_ip_diff'] / Bids_Summary['auc_dev_country_diff']
    Bids_Summary['url_country_ratio'] = Bids_Summary['auc_dev_url_diff'] / Bids_Summary['auc_dev_country_diff']
    print("done")

    #Sort the DS to find out winner, time difference etc
    Bids_DS2 = Bids_DS.ix[:,:'no_of_bids_auction']
    Bids_DS_Sorted =  Bids_DS2.sort(['auction','time'], ascending=True).reset_index(drop=True)

    row_iterator = Bids_DS_Sorted.iterrows()
    before = next(row_iterator)[1]
    #first bid
    before[12]   =  1

    #cost of auction
    before[13]   = 1
    print("starting iteration")

    for i, row in row_iterator:

        #win
        before[9] =  np.where(row[2] == before[2],0,1)
        #Time diff
        row[10]   = np.where(row[2] == before[2],(row[5]-before[5]),0)
        #consecutive bid
        row[11]   = np.where((row[1] == before[1])&(row[2] == before[2]),1,0)
        #first bid
        row[12]   =  np.where(row[2] == before[2],0,1)
        #cost of auction
        row[13]   = np.where(row[2] == before[2],before[13]+1,1)

        before[14]  = np.where(before[9] == 1,before[13],0)

        # Bids_DS_Sorted.loc[i-1, 'win']     =  np.where(row[2] == before[2],0,1)
        # Bids_DS_Sorted.loc[i, 'first_bid'] =  np.where(row[2] == before[2],0,1)
        # Bids_DS_Sorted.loc[i, 'time_diff']  = np.where(row[2] == before[2],(row[5]-before[5]),0)
        # Bids_DS_Sorted.loc[i, 'consecutive_bid'] = np.where((row[1] == before[1])&(row[2] == before[2]),1,0)

        before = row
        if (i == len(Bids_DS_Sorted)-1):
            row[9] = 1
            row[14] = row[13]

        if i%500000 == 0:
            print(i)

    print("ending iteration")
    Bids_DS_Sorted.to_csv(file_path+'Bids_DS_Sorted.csv')

    Bids_DS_auction = Bids_DS_Sorted.groupby(['bidder_id']).agg({'win': [np.sum]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','win']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')
    print("ending win")

    Bids_DS_auction = Bids_DS_Sorted.groupby(['bidder_id']).agg({'time_diff': [np.sum]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','time_diff']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')
    Bids_Summary['time_diff'] = Bids_Summary['time_diff'] / 1000000000
    Bids_Summary['time_diff_avg'] = Bids_Summary['time_diff'] / Bids_Summary['auction']
    print("ending time diff")

    Bids_DS_auction = Bids_DS_Sorted.groupby(['bidder_id']).agg({'consecutive_bid': [np.sum]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','consecutive_bid']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')
    print("ending consecutive_bid")

    Bids_Summary['consecutive_bid_avg'] = Bids_Summary['consecutive_bid'] / Bids_Summary['auction']

    Bids_DS_auction = Bids_DS_Sorted.groupby(['bidder_id']).agg({'first_bid': [np.sum]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','first_bid']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')
    print("ending first bid")

    Bids_DS_auction = Bids_DS_Sorted.groupby(['bidder_id']).agg({'win_cost_of_auction': [np.sum]}).reset_index()
    Bids_DS_auction.columns = ['bidder_id','win_cost_of_auction']
    Bids_Summary = Bids_Summary.merge(Bids_DS_auction, on='bidder_id')
    print("ending win_cost_of_auction")

    #Ending summary value for each time val

    #Move Bids_Summary to Bids , merge with train and Actual DS , keep the order of Actual DS
    Bids_DS = Bids_Summary

    print("Bids formatting completed")
    Bids_DS.to_csv(file_path+'Bids_DS_Summary.csv')

    Actual_DS["Order"] =  Actual_DS.index.values
    Train_DS  = Train_DS.merge(Bids_DS, on='bidder_id',how='left')
    Actual_DS = Actual_DS.merge(Bids_DS, on='bidder_id',how='left')
    Train_DS = Train_DS.fillna(0)
    Actual_DS = Actual_DS.fillna(0)
    print("ending merge")

    Actual_DS = Actual_DS.sort('Order', ascending=1).set_index("Order")
    Actual_DS = Actual_DS.reset_index()
    Actual_DS = Actual_DS.drop(['Order'], axis = 1)

    y = Train_DS.outcome.values

    Train_DS.to_csv(file_path+'Train_New6.csv',index_label='id')
    Actual_DS.to_csv(file_path+'Test_New6.csv',index_label='id')
    Bids_DS.to_csv(file_path+'Bids_New6.csv',index_label='id')
    Sample_DS.to_csv(file_path+'Sample_New5.csv',index_label='id')

    print("***************Ending Data cleansing***************")

    return Train_DS, Actual_DS, y


########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)
    pd.options.mode.chained_assignment = None  # default='warn'

    global file_path
    #file_path = 'C:/Python/Others/data/Kaggle/Facebook_Human_or_Robot/'
    file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Facebook_Human_or_Robot/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS =  pd.read_csv(file_path+'train.csv',sep=',')
    Actual_DS =  pd.read_csv(file_path+'test.csv',sep=',')
    Sample_DS = pd.read_csv(file_path+'sampleSubmission.csv',sep=',')
    Bids_DS = pd.read_csv(file_path+'bids.csv',sep=',')

    # test  = pd.merge(Actual_DS,Bids_DS, on='bidder_id')
    # test_result = test.groupby(['bidder_id'])
    # #Temp = pd.DataFrame(test_result.count())
    # Temp = pd.DataFrame(test_result.merchandise.unique()).reset_index()
    # print(Temp)
    # sys.exit(0)
    #
    # test = test[test['outcome']==1]
    # test_result = test.groupby(['merchandise'])
    # Temp = pd.DataFrame(test_result.count())
    # print(Temp)

    #test.to_csv(file_path+'Bids_DS_Merged_with_Train.csv')
    #sys.exit(0)

    ####################################################################################################################
    #Reduce Bids data for data munging
    # Bids_DS1 = Bids_DS
    # rows = np.random.choice(Bids_DS1.index.values, 5000)
    # Bids_DS = Bids_DS1.ix[rows]
    ###################################################################################################################
    print(np.shape(Bids_DS))
    Train_DS, Actual_DS, y =  Data_Munging(Train_DS, Actual_DS, Bids_DS, Sample_DS)


########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)

