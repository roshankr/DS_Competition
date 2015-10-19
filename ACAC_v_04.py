# -*- coding: utf-8 -*-
import requests
import numpy as np
import scipy as sp
import sys
import platform
import pandas as pd
from time import time
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit , KFold
import re
import time as tm
import warnings
from math import sqrt, exp, log
from csv import DictReader
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier ,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression ,SGDClassifier
from scipy.stats import randint as sp_randint
import sqlite3
from pandas.io import sql
#import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.grid_search import ParameterGrid
from random import randint
import collections
import ast
import re
import demjson
import json
from math import exp, log, sqrt

########################################################################################################################
#AvitoSearch Results Relevance
########################################################################################################################
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
########################################################################################################################
#ftrl class - Follow the Regularized Leader
########################################################################################################################
class ftrl(object):
        def __init__(self, alpha, beta, l1, l2, bits, interaction):
                self.z = [0.] * bits
                self.n = [0.] * bits
                self.alpha = alpha
                self.beta = beta
                self.l1 = l1
                self.l2 = l2
                self.w = {}
                self.X = []
                self.y = 0.
                self.bits = bits
                self.interaction = interaction
                self.Prediction = 0.

        def sgn(self, x):
                if x < 0:
                        return -1
                else:
                        return 1

        #A helper generator that yields the indices in x
        #The purpose of this generator is to make the following
        #code a bit cleaner when doing feature interaction.

        def _indices(self, X):

            # first yield index of the bias term
            yield 0

            # then yield the normal indices
            for index in X:
                yield index

            # now yield interactions (if applicable)
            if self.interaction:
                bits = self.bits
                L = len(X)

                X = sorted(X)
                for i in range(L):
                    for j in range(i+1, L):
                        # one-hot encode interactions with hash trick
                        yield abs(hash(str(X[i]) + '_' + str(X[j]))) % bits

        #Apply hash-trick to the original data,and for simplicity, we one-hot-encode everything
        def fit(self,line):
                # try:
                #         self.ID = line['ID']
                #         del line['ID']
                # except:
                #         pass

                try:
                        self.y = float(line['ISCLICK'])
                        del line['ISCLICK']
                except:
                        pass

                del line['SEARCHMONTH']
                del line['SEARCHYEAR']
                del line['SEARCHDAY']
                del line['LAST_USER_ENTRY']

                self.X = [0.] * len(line)
                for i, key in enumerate(line):
                        val = line[key]
                        self.X[i] = (abs(hash(key + '_' + str(val))) % self.bits)
                self.X = [0] + self.X

        #Bounded logloss
        #logarithmic loss of p given y
        def logloss(self):
                act = self.y
                pred = self.Prediction
                predicted = max(min(pred, 1. - 10e-15), 10e-15)
                return -log(predicted) if act == 1. else -log(1. - predicted)

        #Get probability estimation on x
        #INPUT: x: features , OUTPUT : probability of p(y = 1 | x; w)
        def predict(self):
                W_dot_x = 0.
                w = {}
                for i in self._indices(self.X):
                        if abs(self.z[i]) <= self.l1:
                                w[i] = 0.
                        else:
                                w[i] = (self.sgn(self.z[i]) * self.l1 - self.z[i]) / (((self.beta + sqrt(self.n[i]))/self.alpha) + self.l2)
                        W_dot_x += w[i]
                self.w = w
                self.Prediction = 1. / (1. + exp(-max(min(W_dot_x, 35.), -35.)))
                return self.Prediction

        #Update Model
        # Increases self.n: increase by squared gradient , self.z: weights
        def update(self, prediction):
                for i in self._indices(self.X):
                        g = (prediction - self.y) #* i
                        sigma = (1./self.alpha) * (sqrt(self.n[i] + g*g) - sqrt(self.n[i]))
                        self.z[i] += g - sigma*self.w[i]
                        self.n[i] += g*g

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
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid(X, y, clf):

    print("***************Starting Nfold Cross validation***************")

    X =np.array(X)
    y = np.array(y)

    scores=[]
    #ss = StratifiedShuffleSplit(y, n_iter=5,test_size=0.3, random_state=42, indices=None)
    ss = KFold(len(y), n_folds=5,shuffle=False,indices=False)
    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        clf.fit(X_train, y_train)
        y_pred=clf.predict_proba(X_test)
        scores.append(log_loss(y_test,y_pred, eps=1e-15, normalize=True ))
        print(" %d-iteration... %s " % (i,scores))
        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))

    print("***************Ending Nfold Cross validation***************")

    return scores

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################

def Findall_Keys(SearchParams):
    return re.findall(r'\b\d+(?=:)\b',SearchParams)

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Parm_Matching(SearchParams_DS,AdsParams_DS):

    print("***************Starting Parms matching***************")

    test_DF = pd.DataFrame()

    SearchParams = "{797:'16', 709:'Диски', 799:'6', 798:'7', 5:'Шины, диски и колёса', 801:{to:'-15', from:'-15'}, 800:'139.7'}"

    test_DF.loc[0, 'search'] = SearchParams
    test_DF.loc[1, 'search'] = "{797:'16', 709:'Диски', 799:'6', 798:'7', 5:'Шины, диски и колёса', 801:{to:'-15', from:'-15'},8000:'139.7'}"
    test_DF.loc[2, 'search'] = "{0}"

    test_DF['search'] = test_DF['search'].apply(Findall_Keys)

    print(test_DF)
    print((len(set(test_DF['search'][0]) & set(test_DF['search'][1]))))
    #http://stackoverflow.com/questions/16353729/pandas-how-to-use-apply-function-to-multiple-columns

    #SearchParams = re.sub(':', '11111',str(SearchParams))
    #print(SearchParams)
    #Keys_S = re.findall(r'\b(\d+11111)\b',SearchParams)

    #Keys_S = re.findall(r'\b\d+(?=:)\b',SearchParams)

    #Keys_A = re.findall(r'\b(\w+11111)\b',SearchParams)

    #test = [int(x.group()) for x in re.finditer(r'\d+', SearchParams)]

    #Match = (len(set(Keys_S) & set(Keys_A)))

    Keys_S = Findall_Keys(SearchParams)

    #print(Keys_S)
    sys.exit(0)

    #SearchParams = json.dumps(SearchParams)

    #json1_data = json.loads(SearchParams)

    SearchParams = re.sub('[!@#${}to]', '',str(SearchParams))

    print(SearchParams)

    SearchParams = ast.literal_eval(SearchParams)
    print(json1_data)
    Keys_S    = list(SearchParams.keys())
    print(Keys_S)
    sys.exit(0)

    datapoints = json1_data['datapoints']

    print()
    SearchParams = ast.literal_eval(SearchParams)

    print(SearchParams)
    sys.exit(0)
    test = demjson.decode(SearchParams)
    test1 = json.dumps(SearchParams)
    print(test1)
    newDict = {key: 0 for key in SearchParams}
    print(newDict)

    sys.exit(0)

    #SearchParams = ast.literal_eval(SearchParams)
    SearchParams = re.sub('[!@#${}to]', '',str(SearchParams))
    print(SearchParams)

    Keys_A    = list(SearchParams.keys())

    print(SearchParams)
    print(Keys_A)
    sys.exit(0)

    SearchParams_DS = np.array(SearchParams_DS)
    AdsParams_DS    = np.array(AdsParams_DS)

    Matches = []
    Matching_percentage = []

    for row in range(len(SearchParams_DS)):

        SearchParams =  SearchParams_DS[row]
        AdsParams = AdsParams_DS[row]

        print(row)
        if type(SearchParams) is int:
            Keys_S    = [0]
        else:

            SearchParams = ast.literal_eval(SearchParams)
            SearchParams = re.sub('[!@#${}]', '',str(SearchParams))
            SearchParams = ast.literal_eval("{"+SearchParams+"}")

            Keys_S    = list(SearchParams.keys())

        if  type(AdsParams) is int:
            Keys_A    = [0]
        else:
            AdsParams = ast.literal_eval(AdsParams)
            AdsParams = re.sub('[!@#${}]', '',str(AdsParams))
            AdsParams = ast.literal_eval("{"+AdsParams+"}")

            Keys_A    = list(AdsParams.keys())

        Matches.append(len(set(Keys_S) & set(Keys_A)))
        Matching_percentage.append((len(set(Keys_S) & set(Keys_A))) / float(len(Keys_S)))

    print("***************Ending Parms matching***************")
    sys.exit(0)
    return np.array(Matches) , np.array(Matching_percentage)

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging_dummy(Sample_DS):

    print("***************Starting Data cleansing***************")
    #total count of ObjectType = 3 is 190157735
    conn = sqlite3.connect(file_path+'database.sqlite')

    with conn:
         cur = conn.cursor()
         # cur.execute("CREATE INDEX trainSearchStream_click ON trainSearchStream (IsClick);")
         # print("trainSearchStream_click id completed")

         print("start indexing")
         cur.execute("CREATE INDEX PHONEREQUESTSSTREAM_IDX2 ON PHONEREQUESTSSTREAM (UserID,IPID,AdID);")
         print("PHONEREQUESTSSTREAM_IDX id completed")

         cur.execute("CREATE INDEX VISITSSTREAM_IDX2 ON VISITSSTREAM (UserID,IPID,AdID);")
         print("VISITSSTREAM_IDX id completed")

    #     # cur.execute("CREATE INDEX trainSearchStream_Search ON trainSearchStream (SearchID);")
    #     # print("trainSearchStream_Search id completed")
    #     # cur.execute("CREATE INDEX testSearchStream_Search ON testSearchStream (SearchID);")
    #     # print("testSearchStream_Search id completed")
    #     # cur.execute("CREATE INDEX SearchInfo_Search ON SearchInfo (SearchID);")
    #     # print("SearchInfo_Search id completed")
    #     # cur.execute("CREATE INDEX AdsInfo_idx ON AdsInfo (AdID);")
    #     # print("AdsInfo_idx id completed")
    #     # cur.execute("CREATE INDEX UserInfo_idx ON UserInfo (UserID);")
    #     # print("UserInfo_idx id completed")
    #
    #     cur.execute("CREATE INDEX trainSearchStream_Ad ON trainSearchStream (AdID);")
    #     print("trainSearchStream_Search id completed")
    #     cur.execute("CREATE INDEX testSearchStream_Ad ON testSearchStream (AdID);")
    #     print("testSearchStream_Search id completed")
    #     cur.execute("CREATE INDEX VisitsStream_Idx ON VisitsStream (UserID);")
    #     print("VisitsStream_Idx id completed")
    #     cur.execute("CREATE INDEX PhoneRequestsStream_Idx ON PhoneRequestsStream (UserID);")
    #     print("PhoneRequestsStream_Idx id completed")
    #
         # cur.execute("CREATE INDEX trainSearchStream_ObjectType ON trainSearchStream (ObjectType);")
         # print("trainSearchStream_ObjectType id completed")
         # cur.execute("CREATE INDEX testSearchStream_ObjectType ON testSearchStream (ObjectType);")
         # print("testSearchStream_ObjectType id completed")
         # cur.execute("CREATE INDEX SearchInfo_User ON SearchInfo (UserID);")
         # print("SearchInfo_User id completed")
         # cur.execute("CREATE INDEX AdsInfo_loc ON AdsInfo (LocationID);")
         # print("AdsInfo_loc id completed")
         # cur.execute("CREATE INDEX AdsInfo_cat ON AdsInfo (CategoryID);")
         # print("AdsInfo_cat id completed")

         # cur.execute("CREATE INDEX SearchInfo_Loc ON SearchInfo (LocationID);")
         # print("SearchInfo_Loc id completed")
         # cur.execute("CREATE INDEX SearchInfo_Cat ON SearchInfo (CategoryID);")
         # print("SearchInfo_Cat id completed")

    sys.exit(0)
    # Get train data
    # query = """
    # select * from testSearchStream limit 1000000, 100;
    # """
    # XXX = sql.read_sql(query, conn)
    # print(XXX)
    # sys.exit(0)
    # Get train data
    # query = """
    # select train.AdID,train.Position,train.HistCTR,train.IsClick,search.UserID,search.IPID,search.IsUserLoggedOn
    #   from trainSearchStream as train JOIN SearchInfo as search
    #     ON train.SearchID = search.SearchID
    #   where ObjectType = 3
    #   limit 10000;
    # """

    print("***************Ending Data cleansing***************")

    return Sample_DS

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging_old(Sample_DS):

    print("***************Starting Data cleansing*************** at Time: %s" %(tm.strftime("%H:%M:%S")))
    #total count of ObjectType = 3 is 190157735
    #conn = sqlite3.connect('/home/roshan/Downloads/database.sqlite')
    conn = sqlite3.connect(file_path+'database.sqlite')

    with conn:
         cur = conn.cursor()

         ###############################################################################################################
         # Get user click count , whether clicked or not
         ###############################################################################################################
         query_user_count = """
         CREATE TABLE query_user_count AS
            SELECT SEARCH.USERID                     AS USERID
                 , TRAIN.ISCLICK                     AS ISCLICK
                 , COUNT (*)                         AS USER_CLICK_COUNT
              FROM SEARCHINFO SEARCH
                 , TRAINSEARCHSTREAM TRAIN
             WHERE TRAIN.SEARCHID = SEARCH.SEARCHID
            GROUP BY SEARCH.USERID , TRAIN.ISCLICK
         """

         cur.execute(query_user_count)
         print("query_user_count table completed")

         cur.execute("CREATE INDEX query_user_count_idx1 ON query_user_count (USERID);")
         print("query_user_count_idx1 completed")

         cur.execute("CREATE INDEX query_user_count_idx2 ON query_user_count (USERID,ISCLICK);")
         print("query_user_count_idx2 completed")
         ###############################################################################################################
         # Get ad click count , how many times clicked or not clicked
         ###############################################################################################################
         query_ad_count = """
         CREATE TABLE query_ad_count AS
            SELECT TRAIN.ADID                       AS ADID
                 , TRAIN.ISCLICK                    AS ISCLICK
                 , COUNT (*)                        AS AD_CLICK_COUNT
              FROM TRAINSEARCHSTREAM TRAIN
            GROUP BY TRAIN.ADID , TRAIN.ISCLICK
         """

         cur.execute(query_ad_count)
         print("query_ad_count table completed")

         cur.execute("CREATE INDEX query_ad_count_idx1 ON query_ad_count (ADID);")
         print("query_ad_count_idx1 completed")

         cur.execute("CREATE INDEX query_ad_count_idx2 ON query_ad_count (ADID,ISCLICK);")
         print("query_ad_count_idx2 completed")

         ###############################################################################################################
         # Get pos click count , how many times clicked or not clicked
         ###############################################################################################################
         query_pos_count = """
         CREATE TABLE query_pos_count AS
           SELECT TRAIN.POSITION                   AS POSITION
                , TRAIN.ISCLICK                    AS ISCLICK
                , COUNT (*)                        AS POS_COUNT
             FROM TRAINSEARCHSTREAM TRAIN
            GROUP BY TRAIN.POSITION , TRAIN.ISCLICK
         """

         cur.execute(query_pos_count)
         print("query_pos_count table completed")
         ###############################################################################################################
         # Get user ,position, click count , how many times clicked or not clicked
         ###############################################################################################################

         query_user_pos_count = """CREATE TABLE query_user_pos_count AS
                                 SELECT SEARCH.USERID            AS USERID
                                      , TRAIN.POSITION           AS POSITION
                                      , TRAIN.ISCLICK            AS ISCLICK
                                      , COUNT (*)                AS USER_POS_COUNT
                                   FROM TRAINSEARCHSTREAM TRAIN , SEARCHINFO SEARCH
                                  WHERE TRAIN.SEARCHID = SEARCH.SEARCHID
                               GROUP BY SEARCH.USERID , TRAIN.POSITION , TRAIN.ISCLICK """

         cur.execute(query_user_pos_count)
         print("query_user_pos_count table completed")

         cur.execute("CREATE INDEX query_user_pos_count_idx1 ON query_user_pos_count (USERID);")
         print("query_user_pos_count_idx1 completed")

         cur.execute("CREATE INDEX query_user_pos_count_idx2 ON query_user_pos_count (USERID , POSITION);")
         print("query_user_pos_count_idx2 completed")
         ###############################################################################################################
         # Get ad ,position, click count , how many times clicked or not clicked
         ###############################################################################################################
         query_ad_pos_count = """
                                 CREATE TABLE query_ad_pos_count AS
                                 SELECT TRAIN.ADID                       AS ADID
                                      , TRAIN.POSITION                   AS POSITION
                                      , TRAIN.ISCLICK                    AS ISCLICK
                                      , COUNT (*)                        AS AD_POS_COUNT
                                   FROM TRAINSEARCHSTREAM TRAIN
                                 GROUP BY TRAIN.ADID , TRAIN.POSITION , TRAIN.ISCLICK
                               """

         cur.execute(query_ad_pos_count)
         print("query_ad_pos_count table completed")

         cur.execute("CREATE INDEX query_ad_pos_count_idx1 ON query_ad_pos_count (ADID);")
         print("query_ad_pos_count_idx1 completed")

         cur.execute("CREATE INDEX query_ad_pos_count_idx2 ON query_ad_pos_count (ADID , POSITION);")
         print("query_ad_pos_count_idx2 completed")


    sys.exit(0)

    # query = """
    #     SELECT * FROM SearchInfo B
    #    limit 100
    #
    # """
    #
    # Train_DS = sql.read_sql(query, conn)
    # print(Train_DS)
    #
    # sys.exit(0)

    # query = """
    # select COALESCE(NULLIF(train.SearchID,''), 0) as SearchID
    #       ,COALESCE(NULLIF(train.AdID,''), 0)     as AdID
    #       ,COALESCE(NULLIF(train.Position,''), 0) as Position
    #       ,COALESCE(NULLIF(train.ObjectType,''), 0) as ObjectType
    #       ,COALESCE(NULLIF(train.HistCTR,''), 0) as HistCTR
    #       ,COALESCE(NULLIF(train.IsClick,''), 0) as IsClick
    #       ,COALESCE(NULLIF(strftime('%m', SearchDate),''), 0) as Searchmonth
    #       ,COALESCE(NULLIF(strftime('%Y', SearchDate),''), 0) as Searchyear
    #       ,COALESCE(NULLIF(strftime('%d', SearchDate),''), 0) as Searchday
    # 	  ,COALESCE(NULLIF(search.IPID,''), 0) as IPID
    # 	  ,COALESCE(NULLIF(search.UserID,''), 0) as UserID
    # 	  ,COALESCE(NULLIF(search.IsUserLoggedOn,''), 0) as IsUserLoggedOn
    #       ,COALESCE(NULLIF(search.LocationID,''), 0) as SearchLocationID
    #       ,COALESCE(NULLIF(search.CategoryID,''), 0) as SearchCategoryID
    #
    #       ,COALESCE(NULLIF(user.UserAgentID,''), 0) as UserAgentID/home/roshan/anaconda3/envs/python2/bin/python /home/roshan/Desktop/DS/Python/Avito_Context_Ad_Clicks_v_02.py
    #       ,COALESCE(NULLIF(user.UserAgentFamilyID,''), 0) as UserAgentFamilyID
    #       ,COALESCE(NULLIF(user.UserAgentOSID,''), 0) as UserAgentOSID
    #       ,COALESCE(NULLIF(user.UserDeviceID,''), 0) as UserDeviceID
    #
    #       ,COALESCE(NULLIF(ads.Price,''), 0) as Price
    #       ,COALESCE(NULLIF(ads.IsContext,''), 0) as IsContext
    #
    #       ,COALESCE(NULLIF(category.Level,''), 0) as CLevel
    #       ,COALESCE(NULLIF(category.ParentCategoryID,''), 0) as ParentCategoryID
    #       ,COALESCE(NULLIF(category.SubcategoryID,''), 0) as SubcategoryID
    #
    #
    #       ,COALESCE(NULLIF(location.Level,''), 0) as LLevel
    #       ,COALESCE(NULLIF(location.RegionID,''), 0) as RegionID
    #       ,COALESCE(NULLIF(location.CityID,''), 0) as CityID
    #
    #   from trainSearchStream as train
    #   LEFT JOIN AdsInfo as ads
    #    ON train.AdID = ads.AdID
    #   LEFT JOIN SearchInfo as search
    #    ON train.SearchID = search.SearchID
    #   LEFT JOIN UserInfo as user
    #    ON search.UserID = user.UserID
    #   LEFT JOIN Category as category
    #    ON search.CategoryID = category.CategoryID
    #   LEFT JOIN Location as location
    #    ON search.LocationID = location.LocationID
    #
    #   where ObjectType = 3
    #   LIMIT 1000000,3000000
    # """

    phone_query = """
        SELECT USERID AS USERID
              ,IPID   AS IPID
              ,ADID   AS ADID
              ,COUNT(*) AS COUNT_PHONE
          FROM PHONEREQUESTSSTREAM
        GROUP BY USERID,IPID,ADID
    """
    Phone_DS = sql.read_sql(phone_query, conn)
    print("phone query completed....... at Time: %s" %(tm.strftime("%H:%M:%S")))

    # visit_query = """
    #     SELECT USERID AS USERID
    #           ,IPID   AS IPID
    #           ,ADID   AS ADID
    #           ,COUNT(*) AS VISIT_COUNT
    #       FROM VISITSSTREAM
    #     GROUP BY USERID,IPID,ADID
    # """
    # Visit_DS = sql.read_sql(visit_query, conn)
    # print(Visit_DS.head())
    # print("visit query completed....... at Time: %s" %(tm.strftime("%H:%M:%S")))

    query = """
   SELECT TRAIN_SEARCH.SEARCHID
         ,TRAIN_SEARCH.ADID
         ,TRAIN_SEARCH.POSITION
         ,TRAIN_SEARCH.HISTCTR
         ,TRAIN_SEARCH.ISCLICK
         ,TRAIN_SEARCH.SEARCHMONTH
         ,TRAIN_SEARCH.SEARCHYEAR
         ,TRAIN_SEARCH.SEARCHDAY
         ,TRAIN_SEARCH.IPID
         ,TRAIN_SEARCH.USERID
         ,TRAIN_SEARCH.ISUSERLOGGEDON
         ,TRAIN_SEARCH.SEARCHLOCATIONID
         ,TRAIN_SEARCH.SEARCHCATEGORYID
         ,TRAIN_SEARCH.SEARCHPARAMS

         ,COALESCE(NULLIF(USER.USERAGENTID,''), 0)          AS USERAGENTID
         ,COALESCE(NULLIF(USER.USERAGENTFAMILYID,''), 0)    AS USERAGENTFAMILYID
         ,COALESCE(NULLIF(USER.USERAGENTOSID,''), 0)        AS USERAGENTOSID
         ,COALESCE(NULLIF(USER.USERDEVICEID,''), 0)         AS USERDEVICEID

         ,COALESCE(NULLIF(ADS.PRICE,''), 0)                 AS ADSPRICE
         ,COALESCE(NULLIF(ADS.PARAMS,''), 0)                AS ADSPARAMS

     FROM
  ( SELECT COALESCE(NULLIF(TRAIN.SEARCHID,''), 0) AS SEARCHID
          ,COALESCE(NULLIF(TRAIN.ADID,''), 0)     AS ADID
          ,COALESCE(NULLIF(TRAIN.POSITION,''), 0) AS POSITION
          ,COALESCE(NULLIF(TRAIN.HISTCTR,''), 0) AS HISTCTR
          ,COALESCE(NULLIF(TRAIN.ISCLICK,''), 0) AS ISCLICK

          ,COALESCE(NULLIF(STRFTIME('%m', SEARCHDATE),''), 0) AS SEARCHMONTH
          ,COALESCE(NULLIF(STRFTIME('%Y', SEARCHDATE),''), 0) AS SEARCHYEAR
          ,COALESCE(NULLIF(STRFTIME('%d', SEARCHDATE),''), 0) AS SEARCHDAY
    	  ,COALESCE(NULLIF(SEARCH.IPID,''), 0) AS IPID
    	  ,COALESCE(NULLIF(SEARCH.USERID,''), 0) AS USERID
    	  ,COALESCE(NULLIF(SEARCH.ISUSERLOGGEDON,''), 0) AS ISUSERLOGGEDON
    	  ,COALESCE(NULLIF(SEARCH.LOCATIONID,''), 0) AS SEARCHLOCATIONID
    	  ,COALESCE(NULLIF(SEARCH.CATEGORYID,''), 0) AS SEARCHCATEGORYID
    	  ,COALESCE(NULLIF(SEARCH.SEARCHPARAMS,''), 0    ) AS SEARCHPARAMS

      FROM TRAINSEARCHSTREAM_TEMP AS TRAIN
           ,SEARCHINFO AS SEARCH
     WHERE TRAIN.SEARCHID = SEARCH.SEARCHID

      LIMIT 8000000
   ) TRAIN_SEARCH

   LEFT JOIN ADSINFO  AS ADS
          ON TRAIN_SEARCH.ADID = ADS.ADID
   LEFT JOIN USERINFO AS USER
          ON TRAIN_SEARCH.USERID = USER.USERID
    """

    #   AND RANDOM() % 2 = 0
    # LIMIT 50000000

    Train_DS = sql.read_sql(query, conn)
    print("Train fetched.......... at Time: %s" %(tm.strftime("%H:%M:%S")))

    Train_DS = pd.merge(Train_DS,Phone_DS,on=['USERID','IPID','ADID'],how='left')
    #Train_DS = pd.merge(Train_DS,Visit_DS,on=['USERID','IPID','ADID'],how='left')
    Train_DS = Train_DS.fillna(0)

    Train_DS_IsClick = Train_DS.groupby(['ISCLICK'])
    print(Train_DS_IsClick.ISCLICK.count())

    y = Train_DS.ISCLICK
    #Train_DS = Train_DS.drop(['ISCLICK'], axis = 1)
    cols = Train_DS.columns
    Train_DS, y = shuffle(Train_DS, y, random_state=42)

    Train_DS = Train_DS.drop(['SEARCHPARAMS','ADSPARAMS'], axis = 1)
    print(np.shape(Train_DS))
    print("Train data cleaning completed........... at Time: %s" %(tm.strftime("%H:%M:%S")))

    # # Get test data
    # query_test = """
    # select COALESCE(NULLIF(train.SearchID,''), 0) as SearchID
    #       ,COALESCE(NULLIF(train.AdID,''), 0)     as AdID
    #       ,COALESCE(NULLIF(train.Position,''), 0) as Position
    #       ,COALESCE(NULLIF(train.ObjectType,''), 0) as ObjectType
    #       ,COALESCE(NULLIF(train.HistCTR,''), 0) as HistCTR
    #       ,COALESCE(NULLIF(strftime('%m', SearchDate),''), 0) as Searchmonth
    #       ,COALESCE(NULLIF(strftime('%Y', SearchDate),''), 0) as Searchyear
    #       ,COALESCE(NULLIF(strftime('%d', SearchDate),''), 0) as Searchday
    # 	  ,COALESCE(NULLIF(search.IPID,''), 0) as IPID
    # 	  ,COALESCE(NULLIF(search.UserID,''), 0) as UserID
    # 	  ,COALESCE(NULLIF(search.IsUserLoggedOn,''), 0) as IsUserLoggedOn
    #       ,COALESCE(NULLIF(search.LocationID,''), 0) as SearchLocationID
    #       ,COALESCE(NULLIF(search.CategoryID,''), 0) as SearchCategoryID
    #
    #       ,COALESCE(NULLIF(user.UserAgentID,''), 0) as UserAgentID
    #       ,COALESCE(NULLIF(user.UserAgentFamilyID,''), 0) as UserAgentFamilyID
    #       ,COALESCE(NULLIF(user.UserAgentOSID,''), 0) as UserAgentOSID
    #       ,COALESCE(NULLIF(user.UserDeviceID,''), 0) as UserDeviceID
    #
    #       ,COALESCE(NULLIF(ads.Price,''), 0) as Price
    #       ,COALESCE(NULLIF(ads.IsContext,''), 0) as IsContext
    #
    #       ,COALESCE(NULLIF(category.Level,''), 0) as CLevel
    #       ,COALESCE(NULLIF(category.ParentCategoryID,''), 0) as ParentCategoryID
    #       ,COALESCE(NULLIF(category.SubcategoryID,''), 0) as SubcategoryID
    #
    #
    #       ,COALESCE(NULLIF(location.Level,''), 0) as LLevel
    #       ,COALESCE(NULLIF(location.RegionID,''), 0) as RegionID
    #       ,COALESCE(NULLIF(location.CityID,''), 0) as CityID
    #
    #   from testSearchStream as train
    #   LEFT JOIN AdsInfo as ads
    #    ON train.AdID = ads.AdID
    #   LEFT JOIN SearchInfo as search
    #    ON train.SearchID = search.SearchID
    #   LEFT JOIN UserInfo as user
    #    ON search.UserID = user.UserID
    #   LEFT JOIN Category as category
    #    ON search.CategoryID = category.CategoryID
    #   LEFT JOIN Location as location
    #    ON search.LocationID = location.LocationID
    #
    #   where ObjectType = 3
    # """

    # query_test = """
    # select Position, HistCTR from testSearchStream where ObjectType = 3
    # """

    query_test = """
   SELECT TRAIN_SEARCH.SEARCHID                              AS SEARCHID
         ,TRAIN_SEARCH.ADID                                  AS ADID
         ,TRAIN_SEARCH.POSITION                              AS POSITION
         ,TRAIN_SEARCH.HISTCTR                               AS HISTCTR
         ,TRAIN_SEARCH.SEARCHMONTH                           AS SEARCHMONTH
         ,TRAIN_SEARCH.SEARCHYEAR                            AS SEARCHYEAR
         ,TRAIN_SEARCH.SEARCHDAY                             AS SEARCHDAY
         ,TRAIN_SEARCH.IPID                                  AS IPID
         ,TRAIN_SEARCH.USERID                                AS USERID
         ,TRAIN_SEARCH.ISUSERLOGGEDON                        AS ISUSERLOGGEDON
         ,TRAIN_SEARCH.SEARCHLOCATIONID                      AS SEARCHLOCATIONID
         ,TRAIN_SEARCH.SEARCHCATEGORYID                      AS SEARCHCATEGORYID
         ,TRAIN_SEARCH.SEARCHPARAMS                          AS SEARCHPARAMS
         ,COALESCE(NULLIF(USER.USERAGENTID,''), 0)           AS USERAGENTID
         ,COALESCE(NULLIF(USER.USERAGENTFAMILYID,''), 0)     AS USERAGENTFAMILYID
         ,COALESCE(NULLIF(USER.USERAGENTOSID,''), 0)         AS USERAGENTOSID
         ,COALESCE(NULLIF(USER.USERDEVICEID,''), 0)          AS USERDEVICEID
         ,COALESCE(NULLIF(ADS.PRICE,''), 0)                  AS ADSPRICE
         ,COALESCE(NULLIF(ADS.PARAMS,''), 0)                 AS ADSPARAMS

     FROM
  ( SELECT COALESCE(NULLIF(TRAIN.SEARCHID,''), 0) AS SEARCHID
          ,COALESCE(NULLIF(TRAIN.ADID,''), 0)     AS ADID
          ,COALESCE(NULLIF(TRAIN.POSITION,''), 0) AS POSITION
          ,COALESCE(NULLIF(TRAIN.HISTCTR,''), 0) AS HISTCTR

          ,COALESCE(NULLIF(STRFTIME('%m', SEARCHDATE),''), 0) AS SEARCHMONTH
          ,COALESCE(NULLIF(STRFTIME('%Y', SEARCHDATE),''), 0) AS SEARCHYEAR
          ,COALESCE(NULLIF(STRFTIME('%d', SEARCHDATE),''), 0) AS SEARCHDAY
    	  ,COALESCE(NULLIF(SEARCH.IPID,''), 0) AS IPID
    	  ,COALESCE(NULLIF(SEARCH.USERID,''), 0) AS USERID
    	  ,COALESCE(NULLIF(SEARCH.ISUSERLOGGEDON,''), 0) AS ISUSERLOGGEDON
    	  ,COALESCE(NULLIF(SEARCH.LOCATIONID,''), 0) AS SEARCHLOCATIONID
    	  ,COALESCE(NULLIF(SEARCH.CATEGORYID,''), 0) AS SEARCHCATEGORYID
    	  ,COALESCE(NULLIF(SEARCH.SEARCHPARAMS,''), 0    ) AS SEARCHPARAMS

      FROM TESTSEARCHSTREAM_TEMP AS TRAIN
           ,SEARCHINFO AS SEARCH
     WHERE TRAIN.SEARCHID = SEARCH.SEARCHID
       AND ObjectType = 3

   ) TRAIN_SEARCH

   LEFT JOIN ADSINFO  AS ADS
          ON TRAIN_SEARCH.ADID = ADS.ADID
   LEFT JOIN USERINFO AS USER
          ON TRAIN_SEARCH.USERID = USER.USERID
    """

    Actual_DS = sql.read_sql(query_test, conn)
    print("Test fetched........ at Time: %s" %(tm.strftime("%H:%M:%S")))

    Actual_DS = pd.merge(Actual_DS,Phone_DS,on=['USERID','IPID','ADID'],how='left')
    #Actual_DS = pd.merge(Actual_DS,Visit_DS,on=['USERID','IPID','ADID'],how='left')
    Actual_DS = Actual_DS.fillna(0)

    Actual_DS = Actual_DS.drop(['SEARCHPARAMS','ADSPARAMS'], axis = 1)
    print(np.shape(Actual_DS))
    print("Test data cleaning completed............. at Time: %s" %(tm.strftime("%H:%M:%S")))

    # print("starting Standard Scaler conversion...")
    # # #Setting Standard scaler for data
    # stdScaler = StandardScaler()
    # stdScaler.fit(Train_DS)
    # Train_DS = stdScaler.transform(Train_DS)
    # Actual_DS = stdScaler.transform(Actual_DS)
    #
    # Train_DS = pd.DataFrame(Train_DS,columns=cols)
    # Actual_DS = pd.DataFrame(Actual_DS,columns=cols)

    print("***************Ending Data cleansing*************** at Time: %s" %(tm.strftime("%H:%M:%S")))

    return Train_DS, Actual_DS, y

########################################################################################################################
# Fetch Train Data
########################################################################################################################
def Data_Munging_Train_DS():

    Train_DS = pd.DataFrame()

    periods = 5000000

    for i in range(6):
        dstart = i*periods
        dend   = periods

        d = { 'limit1': dstart, 'limit2': dend }
        conn = sqlite3.connect(file_path+'database.sqlite')
        query = """
            SELECT SEARCHID
                  ,ADID
                  ,POSITION
                  ,HISTCTR
                  ,ISCLICK
    	          ,SEARCHDATE
                  ,SEARCHMONTH
                  ,SEARCHYEAR
                  ,SEARCHDAY
    	          ,IPID
    	          ,USERID
    	          ,ISUSERLOGGEDON
    	          ,SEARCHLOCATIONID
    	          ,SEARCHCATEGORYID
    	          ,SEARCHPARAMS
                  ,USERAGENTID
                  ,USERAGENTFAMILYID
                  ,USERAGENTOSID
                  ,USERDEVICEID
                  ,ADSPRICE
                  ,ADSPARAMS

              FROM TRAINSEARCHSTREAM_FULL
             LIMIT {limit1} , {limit2}
               """
        query = query.format(**d)
        Train_DS1 = sql.read_sql(query, conn)

        conn.close()
        Train_DS = pd.concat([Train_DS,Train_DS1], axis=0)
        print("Fetched records  at Time: %s" %(tm.strftime("%H:%M:%S")))
        print(np.shape(Train_DS))

    return Train_DS

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Sample_DS):

    print("***************Starting Data cleansing*************** at Time: %s" %(tm.strftime("%H:%M:%S")))

    conn = sqlite3.connect(file_path+'database.sqlite')

# ########################################################################################################################
# #  Get user SEARCH count , whether clicked or not
# ########################################################################################################################
#
    # temp_query = """
    #     SELECT USERID
    #          , sum(CASE WHEN ISCLICK=1                      THEN USER_CLICK_COUNT END) AS USER_CNT_ISCLICK_1
    #          , sum(CASE WHEN ISCLICK=0                      THEN USER_CLICK_COUNT END) AS USER_CNT_ISCLICK_0
    #     FROM QUERY_USER_COUNT
    #     GROUP BY USERID
    #
    # """
    #
    # User_Count_DS = (sql.read_sql(temp_query, conn)).fillna(0)
    # User_Count_DS['USER_CNT_ISCLICK_PCT'] = User_Count_DS['USER_CNT_ISCLICK_1'] / (User_Count_DS['USER_CNT_ISCLICK_1'] + User_Count_DS['USER_CNT_ISCLICK_0'])
    # User_Count_DS = User_Count_DS.drop(['USER_CNT_ISCLICK_1','USER_CNT_ISCLICK_0'], axis = 1)
    # print("QUERY_USER_COUNT query completed....... at Time: %s" %(tm.strftime("%H:%M:%S")))
# ########################################################################################################################
# #  Get ad  count , whether clicked or not
# ########################################################################################################################
    temp_query = """
        SELECT ADID
             , sum(CASE WHEN ISCLICK=1                      THEN AD_CLICK_COUNT END) AS AD_CNT_ISCLICK_1
             , sum(CASE WHEN ISCLICK=0                      THEN AD_CLICK_COUNT END) AS AD_CNT_ISCLICK_0
        FROM QUERY_AD_COUNT
        GROUP BY ADID

    """

    Ad_Count_DS = (sql.read_sql(temp_query, conn)).fillna(0)
    Ad_Count_DS['AD_CNT_ISCLICK_PCT'] = Ad_Count_DS['AD_CNT_ISCLICK_1'] / (Ad_Count_DS['AD_CNT_ISCLICK_1'] + Ad_Count_DS['AD_CNT_ISCLICK_0'])
    Ad_Count_DS = Ad_Count_DS.drop(['AD_CNT_ISCLICK_1','AD_CNT_ISCLICK_0'], axis = 1)
    print("QUERY_AD_COUNT query completed....... at Time: %s" %(tm.strftime("%H:%M:%S")))
# # #
# # ########################################################################################################################
# # #  Get position (only 1 and 7)  count , whether clicked or not
# # ########################################################################################################################
    temp_query = """
        SELECT POSITION
             , sum(CASE WHEN ISCLICK=1                      THEN POS_COUNT END) AS POS_CNT_ISCLICK_1
             , sum(CASE WHEN ISCLICK=0                      THEN POS_COUNT END) AS POS_CNT_ISCLICK_0
        FROM QUERY_POS_COUNT
        GROUP BY POSITION
    """

    Pos_Count_DS = (sql.read_sql(temp_query, conn)).fillna(0)
    Pos_Count_DS['POS_CNT_ISCLICK_PCT'] = Pos_Count_DS['POS_CNT_ISCLICK_1'] / (Pos_Count_DS['POS_CNT_ISCLICK_1'] + Pos_Count_DS['POS_CNT_ISCLICK_0'])
    Pos_Count_DS = Pos_Count_DS.drop(['POS_CNT_ISCLICK_1','POS_CNT_ISCLICK_0'], axis = 1)
    print("QUERY_POS_COUNT query completed....... at Time: %s" %(tm.strftime("%H:%M:%S")))

# ########################################################################################################################
# #  Get user / position  count , whether clicked or not
# ########################################################################################################################
    temp_query = """
        SELECT USERID
             , POSITION
             , sum(CASE WHEN ISCLICK=1                      THEN USER_POS_COUNT END) AS USER_POS_CNT_ISCLICK_1
             , sum(CASE WHEN ISCLICK=0                      THEN USER_POS_COUNT END) AS USER_POS_CNT_ISCLICK_0
        FROM QUERY_USER_POS_COUNT
        GROUP BY USERID , POSITION
    """

    User_Pos_Count_DS = (sql.read_sql(temp_query, conn)).fillna(0)
    User_Pos_Count_DS['USER_POS_CNT_ISCLICK_PCT'] = User_Pos_Count_DS['USER_POS_CNT_ISCLICK_1'] / (User_Pos_Count_DS['USER_POS_CNT_ISCLICK_1'] + User_Pos_Count_DS['USER_POS_CNT_ISCLICK_0'])
    User_Pos_Count_DS = User_Pos_Count_DS.drop(['USER_POS_CNT_ISCLICK_1','USER_POS_CNT_ISCLICK_0'], axis = 1)
    print("QUERY_USER_POS_COUNT query completed....... at Time: %s" %(tm.strftime("%H:%M:%S")))
#
# ########################################################################################################################
# #  Get AD / position  count , whether clicked or not
# ########################################################################################################################
    temp_query = """
        SELECT ADID
             , POSITION
             , sum(CASE WHEN ISCLICK=1                      THEN AD_POS_COUNT END) AS AD_POS_CNT_ISCLICK_1
             , sum(CASE WHEN ISCLICK=0                      THEN AD_POS_COUNT END) AS AD_POS_CNT_ISCLICK_0
        FROM QUERY_AD_POS_COUNT
        GROUP BY ADID , POSITION
    """

    Ad_Pos_Count_DS = (sql.read_sql(temp_query, conn)).fillna(0)
    Ad_Pos_Count_DS['AD_POS_CNT_ISCLICK_PCT'] = Ad_Pos_Count_DS['AD_POS_CNT_ISCLICK_1'] / (Ad_Pos_Count_DS['AD_POS_CNT_ISCLICK_1'] + Ad_Pos_Count_DS['AD_POS_CNT_ISCLICK_0'])
    Ad_Pos_Count_DS = Ad_Pos_Count_DS.drop(['AD_POS_CNT_ISCLICK_1','AD_POS_CNT_ISCLICK_0'], axis = 1)
    print("QUERY_AD_POS_COUNT query completed....... at Time: %s" %(tm.strftime("%H:%M:%S")))

# ########################################################################################################################
# # Number of phone requests by user , ipid,adid
# ########################################################################################################################
#     phone_query = """
#         SELECT USERID AS USERID
#               ,IPID   AS IPID
#               ,ADID   AS ADID
#               ,COUNT(*) AS COUNT_PHONE
#           FROM PHONEREQUESTSSTREAM
#         GROUP BY USERID,IPID,ADID
#     """
#     Phone_DS = sql.read_sql(phone_query, conn)
#     print("phone query completed....... at Time: %s" %(tm.strftime("%H:%M:%S")))

########################################################################################################################
# Get last record from train set for each user
########################################################################################################################
    last_user_query = """
        SELECT USERID       AS USERID
              ,SEARCHDATE   AS SEARCHDATE
              ,LAST_USER_ENTRY AS LAST_USER_ENTRY
          FROM TRAINSEARCHSTREAM_LASTUSER
    """
    Last_User_DS = sql.read_sql(last_user_query, conn)
    print("last_user_query completed....... at Time: %s" %(tm.strftime("%H:%M:%S")))

    conn.close()

########################################################################################################################
# Fetch Train Data
########################################################################################################################

    Train_DS = Data_Munging_Train_DS()

    #print(" Parm Match starting" )
    #Matches , Matching_percentage = Parm_Matching(Train_DS['SEARCHPARAMS'],Train_DS['ADSPARAMS'])
    print(np.shape(Train_DS))
    print("Train fetched.......... at Time: %s" %(tm.strftime("%H:%M:%S")))

    #Merge with train and phone

    #Train_DS = pd.merge(Train_DS,Phone_DS,on=['USERID','IPID','ADID'],how='left')
    # Train_DS = pd.merge(Train_DS,User_Count_DS,on=['USERID'],how='left')
    Train_DS = pd.merge(Train_DS,Ad_Count_DS,on=['ADID'],how='left')
    Train_DS = pd.merge(Train_DS,Pos_Count_DS,on=['POSITION'],how='left')
    Train_DS = pd.merge(Train_DS,User_Pos_Count_DS,on=['USERID','POSITION'],how='left')
    Train_DS = pd.merge(Train_DS,Ad_Pos_Count_DS,on=['ADID','POSITION'],how='left')
    Train_DS = pd.merge(Train_DS,Last_User_DS,on=['USERID','SEARCHDATE'],how='left')

    Train_DS2  = Train_DS[Train_DS['LAST_USER_ENTRY']==1]

    Train_DS = Train_DS.fillna(0)
    print(np.shape(Train_DS2))
    print(np.shape(Train_DS))
    print("Train Merge completed.......... at Time: %s" %(tm.strftime("%H:%M:%S")))

    Train_DS_IsClick = Train_DS.groupby(['ISCLICK'])
    print(Train_DS_IsClick.ISCLICK.count())

    y = Train_DS.ISCLICK
    #Train_DS = Train_DS.drop(['ISCLICK'], axis = 1)
    cols = Train_DS.columns
    #Train_DS, y = shuffle(Train_DS, y, random_state=42)

    Train_DS = Train_DS.drop(['SEARCHPARAMS','ADSPARAMS','SEARCHDATE'], axis = 1)
    print(np.shape(Train_DS))
    print(Train_DS.columns)
    print("Train data cleaning completed........... at Time: %s" %(tm.strftime("%H:%M:%S")))

########################################################################################################################
# Fetch Test Data
########################################################################################################################
    conn = sqlite3.connect(file_path+'database.sqlite')
    query_test = """
   SELECT TRAIN_SEARCH.SEARCHID                              AS SEARCHID
         ,TRAIN_SEARCH.ADID                                  AS ADID
         ,TRAIN_SEARCH.POSITION                              AS POSITION
         ,TRAIN_SEARCH.HISTCTR                               AS HISTCTR
         ,COALESCE(NULLIF(SEARCH.SEARCHDATE,''), 0)          AS SEARCHDATE
         ,COALESCE(NULLIF(STRFTIME('%m', SEARCHDATE),''), 0) AS SEARCHMONTH
         ,COALESCE(NULLIF(STRFTIME('%Y', SEARCHDATE),''), 0) AS SEARCHYEAR
         ,COALESCE(NULLIF(STRFTIME('%d', SEARCHDATE),''), 0) AS SEARCHDAY
    	 ,COALESCE(NULLIF(SEARCH.IPID,''), 0)                AS IPID
    	 ,COALESCE(NULLIF(SEARCH.USERID,''), 0)              AS USERID
    	 ,COALESCE(NULLIF(SEARCH.ISUSERLOGGEDON,''), 0)      AS ISUSERLOGGEDON
    	 ,COALESCE(NULLIF(SEARCH.LOCATIONID,''), 0)          AS SEARCHLOCATIONID
    	 ,COALESCE(NULLIF(SEARCH.CATEGORYID,''), 0)          AS SEARCHCATEGORYID
    	 ,COALESCE(NULLIF(SEARCH.SEARCHPARAMS,''), 0    )    AS SEARCHPARAMS

         ,COALESCE(NULLIF(USER.USERAGENTID,''), 0)           AS USERAGENTID
         ,COALESCE(NULLIF(USER.USERAGENTFAMILYID,''), 0)     AS USERAGENTFAMILYID
         ,COALESCE(NULLIF(USER.USERAGENTOSID,''), 0)         AS USERAGENTOSID
         ,COALESCE(NULLIF(USER.USERDEVICEID,''), 0)          AS USERDEVICEID
         ,COALESCE(NULLIF(ADS.PRICE,''), 0)                  AS ADSPRICE
         ,COALESCE(NULLIF(ADS.PARAMS,''), 0)                 AS ADSPARAMS

     FROM
  ( SELECT COALESCE(NULLIF(TRAIN.SEARCHID,''), 0) AS SEARCHID
          ,COALESCE(NULLIF(TRAIN.ADID,''), 0)     AS ADID
          ,COALESCE(NULLIF(TRAIN.POSITION,''), 0) AS POSITION
          ,COALESCE(NULLIF(TRAIN.HISTCTR,''), 0) AS HISTCTR

      FROM TESTSEARCHSTREAM_TEMP AS TRAIN
     WHERE ObjectType = 3

   ) TRAIN_SEARCH

   LEFT JOIN SEARCHINFO AS SEARCH
          ON TRAIN_SEARCH.SEARCHID    = SEARCH.SEARCHID
   LEFT JOIN ADSINFO  AS ADS
          ON TRAIN_SEARCH.ADID        = ADS.ADID
   LEFT JOIN USERINFO AS USER
          ON SEARCH.USERID            = USER.USERID
    """

    Actual_DS = sql.read_sql(query_test, conn)

    print(np.shape(Actual_DS))
    print("Test fetched........ at Time: %s" %(tm.strftime("%H:%M:%S")))
    conn.close()

    # Actual_DS = pd.merge(Actual_DS,Phone_DS,on=['USERID','IPID','ADID'],how='left')
    # Actual_DS = pd.merge(Actual_DS,User_Count_DS,on=['USERID'],how='left')
    Actual_DS = pd.merge(Actual_DS,Ad_Count_DS,on=['ADID'],how='left')
    Actual_DS = pd.merge(Actual_DS,Pos_Count_DS,on=['POSITION'],how='left')
    Actual_DS = pd.merge(Actual_DS,User_Pos_Count_DS,on=['USERID','POSITION'],how='left')
    Actual_DS = pd.merge(Actual_DS,Ad_Pos_Count_DS,on=['ADID','POSITION'],how='left')
    Actual_DS = pd.merge(Actual_DS,Last_User_DS,on=['USERID','SEARCHDATE'],how='left')
    Actual_DS = Actual_DS.fillna(0)

    print(np.shape(Actual_DS))
    print("Test Merge completed.......... at Time: %s" %(tm.strftime("%H:%M:%S")))

    Actual_DS = Actual_DS.drop(['SEARCHPARAMS','ADSPARAMS','SEARCHDATE'], axis = 1)
    print(np.shape(Actual_DS))
    print(Actual_DS.columns)
    print("Test data cleaning completed............. at Time: %s" %(tm.strftime("%H:%M:%S")))

    # print("starting Standard Scaler conversion...")
    # # #Setting Standard scaler for data
    # stdScaler = StandardScaler()
    # stdScaler.fit(Train_DS)
    # Train_DS = stdScaler.transform(Train_DS)
    # Actual_DS = stdScaler.transform(Actual_DS)
    #
    # Train_DS = pd.DataFrame(Train_DS,columns=cols)
    # Actual_DS = pd.DataFrame(Actual_DS,columns=cols)

    print("***************Ending Data cleansing*************** at Time: %s" %(tm.strftime("%H:%M:%S")))

    return Train_DS, Actual_DS, Last_User_DS, y

########################################################################################################################
#LR Classifier
########################################################################################################################
def LR_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting LR Classifier***************")

    t0 = time()

    Train_DS1  = Train_DS[Train_DS['LAST_USER_ENTRY']==0]
    Train_DS2  = Train_DS[Train_DS['LAST_USER_ENTRY']==1]
    Train_DS = Train_DS1
    y = Train_DS['ISCLICK']
    y2 = Train_DS2['ISCLICK']

    Train_DS  = Train_DS.drop(['ISCLICK','LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)
    Train_DS2 = Train_DS2.drop(['ISCLICK','LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)
    Actual_DS = Actual_DS.drop(['LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)

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

        clf = RandomForestClassifier(n_estimators=100,n_jobs=1)

        # run randomized search
        n_iter_search = 20
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = 'log_loss')

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
        #CV: 0.03753620839877235 with only train ds and actual ds (no merging)
        clf = LogisticRegression()
        #Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        clf.fit(Train_DS, y)

    pred_valid = clf.predict_proba(Train_DS2)
    scores = (log_loss(y2,pred_valid, eps=1e-15, normalize=True ))
    print("CV validation is ")
    print(scores)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)
    print("Actual Model predicted")

    # filename = '/home/roshan/Desktop/DS/Others/data/Kaggle/Avito_Context_Ad_Clicks/output/submission.csv'
    # pd.DataFrame({'ID': df_test.TestId, 'IsClick': pred[:, 1]}).to_csv(filename, index=False)

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual[:, 1], index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_LR_Model_1.csv', index_label='ID')

    print("***************Ending LR Classifier***************")
    return pred_Actual

########################################################################################################################
#SGD_Classifier
########################################################################################################################
def SGD_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting SGD Classifier***************")
    t0 = time()

    Train_DS1  = Train_DS[Train_DS['LAST_USER_ENTRY']==0]
    Train_DS2  = Train_DS[Train_DS['LAST_USER_ENTRY']==1]
    Train_DS = Train_DS1
    y = Train_DS['ISCLICK']
    y2 = Train_DS2['ISCLICK']

    Train_DS  = Train_DS.drop(['ISCLICK','LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)
    Train_DS2 = Train_DS2.drop(['ISCLICK','LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)
    Actual_DS = Actual_DS.drop(['LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)

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

        clf = RandomForestClassifier(n_estimators=100,n_jobs=1)

        # run randomized search
        n_iter_search = 20
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = 'log_loss')

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
        clf = SGDClassifier()
        #Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        clf.fit(Train_DS, y)

    pred_valid = clf.predict_proba(Train_DS2)
    scores = (log_loss(y2,pred_valid, eps=1e-15, normalize=True ))
    print("CV validation is ")
    print(scores)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)
    print("Actual Model predicted")

    # filename = '/home/roshan/Desktop/DS/Others/data/Kaggle/Avito_Context_Ad_Clicks/output/submission.csv'
    # pd.DataFrame({'ID': df_test.TestId, 'IsClick': pred[:, 1]}).to_csv(filename, index=False)

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual[:, 1], index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_SGD_Model_1.csv', index_label='ID')

    print("***************Ending SGD Classifier***************")
    return pred_Actual


########################################################################################################################
#RF_Classifier
########################################################################################################################
def RF_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting RF Classifier***************")
    t0 = time()
    Train_DS1  = Train_DS[Train_DS['LAST_USER_ENTRY']==0]
    Train_DS2  = Train_DS[Train_DS['LAST_USER_ENTRY']==1]
    Train_DS = Train_DS1
    y = Train_DS['ISCLICK']
    y2 = Train_DS2['ISCLICK']

    Train_DS  = Train_DS.drop(['ISCLICK','LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)
    Train_DS2 = Train_DS2.drop(['ISCLICK','LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)
    Actual_DS = Actual_DS.drop(['LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)
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

        clf = RandomForestClassifier(n_estimators=100,n_jobs=1)

        # run randomized search
        n_iter_search = 20
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = 'log_loss')

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
        #CV:0.11071998110957326, LB:0.085
        clf = RandomForestClassifier(n_estimators=500,n_jobs=-1)

        #Ownen model for Avazu
        # clf = RandomForestClassifier(n_estimators=32, max_depth=40, min_samples_split=100, min_samples_leaf=10, random_state=0, criterion='entropy',
        #                      max_features=8, verbose = 1, n_jobs=-1, bootstrap=False)


        #Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        clf.fit(Train_DS, y)

    pred_valid = clf.predict_proba(Train_DS2)
    scores = (log_loss(y2,pred_valid, eps=1e-15, normalize=True ))
    print("CV validation is ")
    print(scores)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)
    print("Actual Model predicted")

    # filename = '/home/roshan/Desktop/DS/Others/data/Kaggle/Avito_Context_Ad_Clicks/output/submission.csv'
    # pd.DataFrame({'ID': df_test.TestId, 'IsClick': pred[:, 1]}).to_csv(filename, index=False)

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual[:, 1], index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_RF_Model_1.csv', index_label='ID')

    print("***************Ending RF Classifier***************")
    return pred_Actual

########################################################################################################################
#XGB_Classifier
########################################################################################################################
def XGB_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting XGB Classifier***************")
    t0 = time()
    Train_DS1  = Train_DS[Train_DS['LAST_USER_ENTRY']==0]
    Train_DS2  = Train_DS[Train_DS['LAST_USER_ENTRY']==1]
    Train_DS = Train_DS1
    y = Train_DS['ISCLICK']
    y2 = Train_DS2['ISCLICK']

    Train_DS  = Train_DS.drop(['ISCLICK','LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)
    Train_DS2 = Train_DS2.drop(['ISCLICK','LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)
    Actual_DS = Actual_DS.drop(['LAST_USER_ENTRY','SEARCHMONTH','SEARCHYEAR','SEARCHDAY'], axis = 1)

    if grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {'n_estimators': [50],
                      'max_depth': [10],
                      'min_child_weight': [4],
                      'subsample': [0.5,0.6, 0.7,0.8, 0.9,1],
                      'colsample_bytree': [0.5,0.6, 0.7,0.8, 0.9,1],
                      'silent':[True],
                      'gamma':[1,0.5,0.6,0.7,0.8,0.9]
                     }

        clf = xgb.XGBClassifier(n_estimators=100)

        # run randomized search
        n_iter_search = 20
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = 'log_loss')

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
        #CV:0.0361782124620307, LB:
        #clf = xgb.XGBClassifier(n_estimators=500)

        #CV:0.0361782124620307, LB:
        clf = xgb.XGBClassifier(n_estimators=500)

        #Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        Train_DS =np.array(Train_DS)
        y = np.array(y)
        Actual_DS =np.array(Actual_DS)

        clf.fit(Train_DS, y)

    Train_DS2 =np.array(Train_DS2)
    y2 = np.array(y2)
    pred_valid = clf.predict_proba(Train_DS2)
    scores = (log_loss(y2,pred_valid, eps=1e-15, normalize=True ))
    print("CV validation is ")
    print(scores)

        #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)
    print("Actual Model predicted")

    # filename = '/home/roshan/Desktop/DS/Others/data/Kaggle/Avito_Context_Ad_Clicks/output/submission.csv'
    # pd.DataFrame({'ID': df_test.TestId, 'IsClick': pred[:, 1]}).to_csv(filename, index=False)

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual[:, 1], index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_XGB_Model_2.csv', index_label='ID')

    print("***************Ending XGB Classifier***************")
    return pred_Actual

########################################################################################################################
#FTLR_Classifier
########################################################################################################################
def FTLR_Classifier(Train_DS, y, Actual_DS, Sample_DS, Last_User_DS, grid):

    print("***************Starting FTLR Classifier***************")
    t0 = time()

    loss_count_display =  1000000
    pred_Actual = []

    if grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")
        #Set up parm list for Random Grid search
        param_dist = {
                       'alpha'  : [0.01, 0.1, 0.2, 0.3,0.4, 0.5,0.6, 0.7,0.8,0.9, 1,0.001,10]
                      ,'l1'     : [0.1, 0.3, 0.5, 0.7,0.9,1.0]
                      ,'l2'     : [0.1, 0.3, 0.5, 0.7,0.9,1.0]
                     }

        #Move it to a list
        parm_list = list(ParameterGrid(param_dist))
        num_iter = 100
        FTRL_parm_details = pd.DataFrame(columns=['alpha','l1','l2','loss'])
        #Execute grid search for num_iter times
        for iter in range(num_iter):

            #generate a random number , get the nth dictionary and delete it (no replacement)
            index = randint(0,len(parm_list)-1)
            test_parms = collections.OrderedDict(sorted(parm_list[index].items()))
            del parm_list[index]

            #Set parms for ftrl
            clf = ftrl( alpha = test_parms['alpha'], beta = 1., l1 = test_parms['l1'], l2 = test_parms['l2'], bits = 2**20) #1048576 = 2**20

            loss = 0.
            count = 0
            loss_count = 0
            holdafter = 20
            i = 0
            row_iterator = 0

            row_iterator = Train_DS.iterrows()
            for i, row in row_iterator:

                #dateval =Train_DS['SEARCHDAY'][i]

                line = row.to_dict()
                clf.fit(line)
                pred = clf.predict()

                count += 1

                if (Train_DS.loc[i, 'LAST_USER_ENTRY'] == 1):
                #if (count%10 == 0):
                    loss_count += 1
                    loss += clf.logloss()
                else:
                    clf.update(pred)

                if count % loss_count_display == 0:
                    print ("(seen, loss) : ", (count, loss * 1./loss_count))

            print("iteration completed - ",iter)
            #set test value s(parms and loss to dataframe)
            cols_val = [test_parms['alpha'],test_parms['l1'],test_parms['l2'],(loss * 1./loss_count)]
            FTRL_parm_details.loc[iter] =  cols_val

        FTRL_parm_details = FTRL_parm_details.sort(columns='loss',ascending=True)

        print(FTRL_parm_details)
        FTRL_parm_details.to_csv(file_path+'FTRL_parm_details.csv')
        sys.exit(0)
    else:
        clf = ftrl( alpha = 0.3,
                beta = 1.,
                l1 = 0.1,
                l2 = 0.1,
                bits = 2**20,
                interaction=True)

        loss = 0.
        count = 0
        holdafter = 29
        row_iterator = 0
        i = 0
        loss_count = 0

        row_iterator = Train_DS.iterrows()
        for i, row in row_iterator:
        #for t, line in enumerate(DictReader(open(file_path+'train.csv'),delimiter='\t')):
            #dateval =Train_DS['SEARCHDAY'][i]

            line = row.to_dict()
            clf.fit(line)
            pred = clf.predict()

            count += 1

            if (Train_DS.loc[i, 'LAST_USER_ENTRY'] == 1):
            #if (count%10 == 0):
                loss += clf.logloss()
                loss_count += 1
            else:
                clf.update(pred)

            if count % loss_count_display == 0:
                    print ("(seen, loss) : ", (count, loss * 1./loss_count))

        print ("(log loss) : ", (loss * 1./loss_count))
        test_iterator = Actual_DS.iterrows()
        ####################################################################################################################################
        #CV: 0.035216016231262276 , LB: 0.04707 , with only Train DS , Actual DS
        ####################################################################################################################################
        with open(file_path+'temp.csv', 'w') as output:
            #for t, line in enumerate(DictReader(open(file_path+'test.csv'),delimiter='\t')):
            for i, row in test_iterator:
                line = row.to_dict()
                clf.fit(line)
                clf.predict()
                pred_Actual.append(clf.predict())
                #output.write('%s\n' % str(clf.predict()))
                if i % loss_count_display == 0:
                    print ("completed : ", (i))

        pred_Actual=np.array(pred_Actual)
        print(np.shape(pred_Actual))
        # preds = np.array(pd.read_csv(file_path+'temp.csv', header = None))
        # index = Sample_DS.ID.values
        print("Actual Model predicted")

        #Get the predictions for actual data set
        preds = pd.DataFrame(pred_Actual, index=Sample_DS.ID.values, columns=Sample_DS.columns[1:])
        preds.to_csv(file_path+'output/Submission_Roshan_FTLR_Model_6.csv', index_label='ID')

        # Sample_DS['IsClick'] = preds[index]
        # Sample_DS.to_csv(file_path+'output/Submission_Roshan_FTLR_Model_1.csv', index=False)

    print("***************Ending FTLR Classifier***************")
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path

    if(platform.system() == "Windows"):
        file_path = 'C:/Python/Others/data/Kaggle/Avito_Context_Ad_Clicks/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Avito_Context_Ad_Clicks/'

    test = Parm_Matching(1,2)
    sys.exit(0)

    f = open(file_path+'out.txt', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    full_run = True
########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################

    Sample_DS = pd.read_csv(file_path+'sampleSubmission.csv',sep=',')

    Train_DS, Actual_DS, Last_User_DS, y =  Data_Munging(Sample_DS)

    pred_Actual = FTLR_Classifier(Train_DS, y, Actual_DS, Sample_DS, Last_User_DS, grid=False)
    #pred_Actual = LR_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = XGB_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = SGD_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = RF_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)