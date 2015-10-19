import numpy as np
import scipy as sp
import sys
import pandas as pd # pandas
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier ,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression ,SGDClassifier,  SGDClassifier,RidgeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.svm import SVC , OneClassSVM
from time import time
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import randint as sp_randint
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
#from random import shuffle
from sklearn.utils import shuffle
from sklearn.lda import LDA
import datetime
from sklearn.mixture import GMM
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
#import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.covariance import EllipticEnvelope

########################################################################################################################
#Facebook - Human or Robot                                                                                     #
########################################################################################################################
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
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS, Actual_DS, Bids_DS):

    print("***************Starting Data cleansing***************")

    #All Data Munging done in separate program

    # Train_DS = Train_DS.drop(['win','first_bid','win_cost_of_auction','auc_dev_country_diff','country',
    #                           'auc_country','auc_dev_ip_diff','auction','auc_dev_ip_diff','time_diff_avg',
    #                           'time_diff','ip','ip_country_ratio','url','device','country_ip_unique',
    #                           'auc_ip','auc_dev','bidder_id_unique','auc_url','auc_dev_url_diff',
    #                           'bid_ip'], axis = 1)
    #
    # Actual_DS = Actual_DS.drop(['win','first_bid','win_cost_of_auction','auc_dev_country_diff','country',
    #                           'auc_country','auc_dev_ip_diff','auction','auc_dev_ip_diff','time_diff_avg',
    #                           'time_diff','ip','ip_country_ratio','url','device','country_ip_unique',
    #                           'auc_ip','auc_dev','bidder_id_unique','auc_url','auc_dev_url_diff',
    #                           'bid_ip'], axis = 1)

    #already completed test
    # Train_DS = Train_DS.drop(['win','first_bid','win_cost_of_auction','auc_dev_country_diff','country',
    #                           'auc_country','auc_dev_ip_diff','auction','auc_dev_ip_diff','time_diff_avg',
    #                           'time_diff','ip','ip_country_ratio','url','device','country_ip_unique',
    #                           'auc_ip','auc_dev_url_diff'], axis = 1)


    Train_DS = Train_DS.drop(['auc_dev','auc_country','auc_ip','auc_url','auc_dev_ip_diff',
                              'auc_dev_url_diff','auc_dev_country_diff','ip_country_ratio','time_diff_avg','consecutive_bid_avg',
                              'win_cost_of_auction','first_bid','win','ip_url_ratio','url_country_ratio'], axis = 1)



    Train_DS['Country_by_device'] = np.where(Train_DS['device'] > 20,Train_DS['country']/Train_DS['device'],0.1)
    Train_DS['ip_by_device'] = np.where(Train_DS['device'] > 0, Train_DS['ip']/Train_DS['device'],0)
    Train_DS['country_by_auction'] = np.where(Train_DS['auction'] > 0, Train_DS['country']/Train_DS['auction'],0)
    Train_DS['url_by_auction'] = np.where(Train_DS['auction'] > 0, Train_DS['url']/Train_DS['auction'],0)
    Train_DS['bid_url']        = np.where(Train_DS['bid_id'] > 0,Train_DS['url']/Train_DS['bid_id'],0)
    Train_DS['country_by_ip'] = np.where(Train_DS['ip'] > 0, Train_DS['country']/Train_DS['ip'],0)

    Train_DS['bid_url']            = np.where(Train_DS['bid_id'] > 0,Train_DS['url']/Train_DS['bid_id'],0)
    Train_DS['country_by_ip']      = np.where(Train_DS['ip'] > 0, Train_DS['country']/Train_DS['ip'],0)
    Train_DS['bid_device']         = np.where(Train_DS['bid_id'] > 100,Train_DS['bid_device'],0)
    Train_DS['bid_country']        = np.where(Train_DS['bid_id'] > 100,Train_DS['bid_country'],0)
    Train_DS['bid_ip']             = np.where(Train_DS['bid_id'] > 100,Train_DS['bid_ip'],0)
    Train_DS['bidder_id_unique']   = np.where(Train_DS['bid_id'] > 100,Train_DS['bidder_id_unique'],0)
    Train_DS['country_ip_unique']  = np.where(Train_DS['bid_id'] > 100,Train_DS['country_ip_unique'],0)
    Train_DS['bid_url']            = np.where(Train_DS['bid_id'] > 100,Train_DS['bid_url'],0)
    Train_DS['Country_by_device']  = np.where(Train_DS['bid_id'] > 100,Train_DS['Country_by_device'],0)
    Train_DS['ip_by_device']       = np.where(Train_DS['bid_id'] > 100,Train_DS['ip_by_device'],0)
    Train_DS['country_by_auction'] = np.where(Train_DS['bid_id'] > 100,Train_DS['country_by_auction'],0)
    Train_DS['url_by_auction']     = np.where(Train_DS['bid_id'] > 100,Train_DS['url_by_auction'],0)

    ###################################################################################################################
    #apply same for Actual DS
    Actual_DS = Actual_DS.drop(['auc_dev','auc_country','auc_ip','auc_url','auc_dev_ip_diff',
                              'auc_dev_url_diff','auc_dev_country_diff','ip_country_ratio','time_diff_avg','consecutive_bid_avg',
                              'win_cost_of_auction','first_bid','win','ip_url_ratio','url_country_ratio'], axis = 1)

    Actual_DS['Country_by_device'] = np.where(Actual_DS['device'] > 20,Actual_DS['country']/Actual_DS['device'],0.1)
    Actual_DS['ip_by_device'] = np.where(Actual_DS['device'] > 0, Actual_DS['ip']/Actual_DS['device'],0)
    Actual_DS['country_by_auction'] = np.where(Actual_DS['auction'] > 0, Actual_DS['country']/Actual_DS['auction'],0)
    Actual_DS['url_by_auction'] = np.where(Actual_DS['auction'] > 0, Actual_DS['url']/Actual_DS['auction'],0)
    Actual_DS['bid_url']        = np.where(Actual_DS['bid_id'] > 0,Actual_DS['url']/Actual_DS['bid_id'],0)
    Actual_DS['country_by_ip'] = np.where(Actual_DS['ip'] > 0, Actual_DS['country']/Actual_DS['ip'],0)

    Actual_DS['bid_url']            = np.where(Actual_DS['bid_id'] > 0,Actual_DS['url']/Actual_DS['bid_id'],0)
    Actual_DS['country_by_ip']      = np.where(Actual_DS['ip'] > 0, Actual_DS['country']/Actual_DS['ip'],0)
    Actual_DS['bid_device']         = np.where(Actual_DS['bid_id'] > 100,Actual_DS['bid_device'],0)
    Actual_DS['bid_country']        = np.where(Actual_DS['bid_id'] > 100,Actual_DS['bid_country'],0)
    Actual_DS['bid_ip']             = np.where(Actual_DS['bid_id'] > 100,Actual_DS['bid_ip'],0)
    Actual_DS['bidder_id_unique']   = np.where(Actual_DS['bid_id'] > 100,Actual_DS['bidder_id_unique'],0)
    Actual_DS['country_ip_unique']  = np.where(Actual_DS['bid_id'] > 100,Actual_DS['country_ip_unique'],0)
    Actual_DS['bid_url']            = np.where(Actual_DS['bid_id'] > 100,Actual_DS['bid_url'],0)
    Actual_DS['Country_by_device']  = np.where(Actual_DS['bid_id'] > 100,Actual_DS['Country_by_device'],0)
    Actual_DS['ip_by_device']       = np.where(Actual_DS['bid_id'] > 100,Actual_DS['ip_by_device'],0)
    Actual_DS['country_by_auction'] = np.where(Actual_DS['bid_id'] > 100,Actual_DS['country_by_auction'],0)
    Actual_DS['url_by_auction']     = np.where(Actual_DS['bid_id'] > 100,Actual_DS['url_by_auction'],0)


    Train_DS = Train_DS.drop(['ip','url','country','time_diff','auction'], axis = 1)
    Actual_DS = Actual_DS.drop(['ip','url','country','time_diff','auction'], axis = 1)

    # Train_DS.to_csv(file_path+'Train_DS_only_counts.csv')
    # sys.exit(0)

    y = Train_DS.outcome.values
    Train_DS = Train_DS.drop(['id','outcome'], axis = 1)
    Actual_DS = Actual_DS.drop(['id'], axis = 1)

    Train_DS = Train_DS.drop(['bidder_id','payment_account','address','merchandise'], axis = 1)
    Actual_DS = Actual_DS.drop(['bidder_id','payment_account','address','merchandise'], axis = 1)

    Train_DS = Train_DS.drop(['country_ip_unique','device','country_by_auction','url_by_auction','Country_by_device',
                              'ip_by_device','consecutive_bid','country_by_ip'], axis = 1)
    Actual_DS = Actual_DS.drop(['country_ip_unique','device','country_by_auction','url_by_auction','Country_by_device',
                              'ip_by_device','consecutive_bid','country_by_ip'], axis = 1)

    ###################################################################################################################
    #Try to include a feature with outlier identifier features
    Train_DS2 = np.array(Train_DS)
    Actual_DS2 = np.array(Actual_DS)

    stdScaler = StandardScaler()
    stdScaler.fit(Train_DS2,y)
    Train_DS2 = stdScaler.transform(Train_DS2)
    Actual_DS2 = stdScaler.transform(Actual_DS2)

    # clf =  EllipticEnvelope(support_fraction=1,contamination=0.051167412)
    # clf.fit(Train_DS2, y)
    #
    # pred_train = clf.predict(Train_DS2)
    # pred_actual = clf.predict(Actual_DS2)
    # Train_DS['EllipticEnvelope'] = np.where(pred_train==1,0,1)
    # Actual_DS['EllipticEnvelope'] = np.where(pred_actual==1,0,1)

    # clf =  OneClassSVM(kernel='rbf',nu=0.95 * 0.051167412 + 0.05,gamma=0.1)
    # clf.fit(Train_DS2, y)
    #
    # pred_train = clf.predict(Train_DS2)
    # pred_actual = clf.predict(Actual_DS2)
    # Train_DS['OneClassSVM'] = np.where(pred_train==1,0,1)
    # Actual_DS['OneClassSVM'] = np.where(pred_actual==1,0,1)


    # pred_train = clf.decision_function(Train_DS2).ravel()
    # pred_actual = clf.decision_function(Actual_DS2).ravel()
    # Train_DS['EllipticEnvelope'] =pred_train
    # Actual_DS['EllipticEnvelope'] = pred_actual

    # clf =  OneClassSVM(kernel='rbf',nu=0.95 * 0.051167412 + 0.05,gamma=0.1)
    # clf.fit(Train_DS2, y)
    #
    # pred_train = clf.decision_function(Train_DS2).ravel()
    # pred_actual = clf.decision_function(Actual_DS2).ravel()
    # Train_DS['OneClassSVM'] =pred_train
    # Actual_DS['OneClassSVM'] = pred_actual

    ###################################################################################################################

    global Train_DS1
    Train_DS1 = Train_DS
    Train_DS, y = shuffle(Train_DS, y, random_state=42)

    Actual_DS.to_csv(file_path+'Actual_DS_only_counts.csv')

    print("***************Ending Data cleansing***************")

    return Train_DS, Actual_DS, y

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid(X, y, clf):

    print("***************Starting Kfold Cross validation***************")

    X = np.array(X)
    scores=[]
    ss = StratifiedShuffleSplit(y, n_iter=50,test_size=0.3, random_state=42, indices=None)
    #ss = KFold(len(y), n_folds=30,shuffle=True,indices=False)
    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        clf.fit(X_train, y_train)
        #y_pred=clf.predict(X_test)
        y_pred=clf.predict_proba(X_test)[:,1]

        scores.append(roc_auc_score(y_test, y_pred))
        print(" %d-iteration... %s " % (i,scores))
        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):
    print("***************Starting Random Forest Classifier***************")
    t0 = time()

    #Train_DS = np.log( 1 + Train_DS)
    #Actual_DS = np.log( 1 + Actual_DS)

    #Setting Standard scaler for data
    # stdScaler = StandardScaler()
    # stdScaler.fit(Actual_DS)
    # Train_DS = stdScaler.transform(Train_DS)
    # Actual_DS = stdScaler.transform(Actual_DS)

    if grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "criterion":['gini', 'entropy'],
                      "max_depth": [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15, None],
                      "max_features": [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15, None,'auto','log2'],
                      "min_samples_split": sp_randint(1, 20),
                      "min_samples_leaf": sp_randint(1, 20),
                      "bootstrap": [True],
                      "oob_score": [True, False]
                     }

        clf = RandomForestClassifier(n_estimators=500,n_jobs=-1)

        # run randomized search
        n_iter_search = 3000
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = 'roc_auc',cv=10)

        start = time()
        clf.fit(Train_DS, y)

        print("RandomizedSearchCV took %.2f seconds for %d candidates"
                " parameter settings." % ((time() - start), n_iter_search))
        report(clf.grid_scores_)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
    else:

        #starting model CV : 0.5354 , LB : 0.73
        #clf = RandomForestClassifier(n_jobs=-1, n_estimators=10, min_samples_split=1)

        #best by Random Search (CV :0.84783 , LB: 0.84613)
        # clf = RandomForestClassifier(n_jobs=-1, n_estimators=500, min_samples_split=6,max_features=6,bootstrap=True,
        #                               max_depth = None, min_samples_leaf = 5)

        #Model with rank: 1 with new features Mean validation score: 0.918 (std: 0.029)
        # clf = RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=11,max_features=3,bootstrap=True,
        #                                max_depth = None, min_samples_leaf = 6)

        #Model with rank: 1 with new features Mean validation score: 0.918 (std: 0.029)
        #clf = RandomForestClassifier(n_jobs=-1, n_estimators=1500, min_samples_split=2,max_features=4,bootstrap=True,
        #                                max_depth = 8, min_samples_leaf = 2)

        #Model with rank: 1 with new features , CV: 0.91728 LB :0.90245
        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        # Model with rank: 1 with new features
        # Mean validation score: 0.918 (std: 0.030)
        # Parameters: {'bootstrap': True, 'max_features': 4, 'min_samples_leaf': 2, 'min_samples_split': 2, 'max_depth': 8}

        #Model with rank: 1 with new features Mean validation score: 0.918 (std: 0.029)
        # clf = RandomForestClassifier(n_jobs=-1, n_estimators=2000, min_samples_split=14,criterion='entropy',
        #                              max_features=6,bootstrap=True,
        #                                 max_depth = 8, min_samples_leaf = 5,oob_score=False)

        #Model with rank: 1 with less features  and removed rows : Mean validation score: 0.918 (std: 0.029)
        clf = RandomForestClassifier(n_jobs=-1, n_estimators=1500, min_samples_split=1,max_features='auto',bootstrap=True,
                                        max_depth = 8, min_samples_leaf = 4,oob_score=True,criterion='entropy')

        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        clf.fit(Train_DS, y)

        feature = pd.DataFrame()
        feature['imp'] = clf.feature_importances_
        feature['col'] = Train_DS1.columns
        feature = feature.sort(['imp'], ascending=False)
        print(feature)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.bidder_id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_RF_305_test.csv', index_label='bidder_id')

    print("***************Ending Random Forest Classifier***************")
    return pred_Actual

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def xgb_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):
    print("***************Starting XGB Classifier***************")
    t0 = time()

    # Train_DS = np.log( 1 + Train_DS)
    # Actual_DS = np.log( 1 + Actual_DS)

    #Setting Standard scaler for data
    # stdScaler = StandardScaler()
    # stdScaler.fit(Train_DS,y)
    # Train_DS = stdScaler.transform(Train_DS)
    # Actual_DS = stdScaler.transform(Actual_DS)

    if grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        # param_grid = {'n_estimators': [50],
        #               'max_depth': [6, 1, 3, 5, 8, 10],
        #               'min_child_weight': [1, 4, 7, 10],
        #               'subsample': [0.1, 0.2,0.3, 0.4,0.5,0.6, 0.7,0.8, 0.9,1],
        #               'colsample_bytree': [0.1, 0.2,0.3, 0.4,0.5,0.6, 0.7,0.8, 0.9,1],
        #               'silent':[True],
        #               'gamma':[1,0.5,0.6,0.7,0.8,0.9]
        #              }

        param_grid = {'n_estimators': [500],
                      'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                      'min_child_weight': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                      'subsample': [0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9,1],
                      'colsample_bytree': [0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9,1],
                      'silent':[True],
                      'gamma':[2,1,0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9]
                     }

        # clf = GridSearchCV(xgb.XGBClassifier(),param_grid, scoring='roc_auc',
        #                    verbose=1,cv=10)

        #run randomized search
        n_iter_search = 3000
        clf = xgb.XGBClassifier(nthread=-1)
        clf = RandomizedSearchCV(clf, param_distributions=param_grid,
                                           n_iter=n_iter_search, scoring = 'roc_auc',cv=10)

        start = time()
        clf.fit(Train_DS, y)

        print("GridSearchCV completed")
        report(clf.grid_scores_)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

    else:

        #starting model
        # clf = xgb.XGBClassifier(n_estimators=200,max_depth=10,learning_rate=0.01,nthread=2,min_child_weight=4,
        #                      subsample=0.9,colsample_bytree=0.8,silent=True, gamma = 1)

        # Model with rank: 1 , Mean validation score: 0.921 (std: 0.024)
        # clf = xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,nthread=2,min_child_weight=1,
        #                      subsample=0.5,colsample_bytree=0.9,silent=True, gamma = 0.6)

        # Model with rank: 1
        # Mean validation score: 0.919 (std: 0.031)
        # Parameters: {'colsample_bytree': 0.8, 'silent': True, 'min_child_weight': 4, 'n_estimators': 200, 'subsample': 1, 'max_depth': 3, 'gamma': 0.8}
        #
        # Model with rank: 2
        # Mean validation score: 0.918 (std: 0.032)
        # Parameters: {'colsample_bytree': 0.8, 'silent': True, 'min_child_weight': 4, 'n_estimators': 200, 'subsample': 1, 'max_depth': 3, 'gamma': 1}
        #
        # Model with rank: 3
        # Mean validation score: 0.918 (std: 0.028)
        # Parameters: {'colsample_bytree': 0.7, 'silent': True, 'min_child_weight': 1, 'n_estimators': 200, 'subsample': 0.6, 'max_depth': 9, 'gamma': 0.6}
        #
        # Best estimator found by grid search:
        # XGBClassifier(base_score=0.5, colsample_bytree=0.8, gamma=0.8,
        # learning_rate=0.1, max_delta_step=0, max_depth=3,
        # min_child_weight=4, n_estimators=200, nthread=-1,
        # objective='binary:logistic', seed=0, silent=True, subsample=1)

        #Cv = .91278 LB : 0.89863
        # clf = xgb.XGBClassifier(n_estimators=1000,max_depth=3,learning_rate=0.1,nthread=2,min_child_weight=4,
        #                      subsample=1,colsample_bytree=0.8,silent=True, gamma = 1)


        clf = xgb.XGBClassifier(n_estimators=1000,max_depth=6,learning_rate=0.1,nthread=2,min_child_weight=1,
                             subsample=0.9,colsample_bytree=1,silent=True, gamma = 0.7)


        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set

    preds = pd.DataFrame(pred_Actual, index=Sample_DS.bidder_id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_xgb_305.csv', index_label='bidder_id')

    print("***************Ending XGB Classifier***************")
    return pred_Actual

########################################################################################################################
#Logistic Regression  Classifier
########################################################################################################################
def LogReg_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):
    print("***************Starting Logit Classifier***************")
    t0 = time()

    Train_DS = np.log( 1 + Train_DS)
    Actual_DS = np.log( 1 + Actual_DS)

    #Setting Standard scaler for data
    stdScaler = StandardScaler()
    stdScaler.fit(Train_DS,y)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

    if grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "penalty": [ 'l1','l2'],
                      "C": [ 0.005,0.01,0.03, 0.05,0.07,0.09, 0.1,0.3,0.5,0.7,0.9,1],
                     }

        clf = LogisticRegression(dual=False)

        clf = GridSearchCV(clf,param_dist, scoring='roc_auc', n_jobs=2,
                           verbose=1,cv=10)

        # run randomized search
        # n_iter_search = 200
        # clf = RandomizedSearchCV(clf, param_distributions=param_dist,
        #                                    n_iter=n_iter_search, scoring = 'roc_auc',cv=10)

        start = time()
        clf.fit(Train_DS, y)

        # print("RandomizedSearchCV took %.2f seconds for %d candidates"
        #         " parameter settings." % ((time() - start), n_iter_search))
        report(clf.grid_scores_)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        print(clf.grid_scores_)
        print(clf.best_score_)
        print(clf.best_params_)
        print(clf.scorer_)
    else:

        #Mean validation score: 0.758 (std: 0.075) - LB 0.7373
        clf = LogisticRegression(C=0.09,penalty='l1')

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set

    preds = pd.DataFrame(pred_Actual, index=Sample_DS.bidder_id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_Logit.csv', index_label='bidder_id')

    print("***************Ending Logit Classifier***************")
    return pred_Actual

########################################################################################################################
#SVM with rbf  Classifier
########################################################################################################################
def SVM_rbf_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting SVM with rbf Classifier***************")
    t0 = time()

    Train_DS = np.log( 1 + Train_DS)
    Actual_DS = np.log( 1 + Actual_DS)

    #Setting Standard scaler for data
    stdScaler = StandardScaler()
    stdScaler.fit(Train_DS,y)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

    if grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      'C': [1e3, 5e3, 1e4, 5e4, 1e5, 5e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
                     }

        clf = SVC(kernel='rbf', class_weight='auto', probability=True)

        clf = GridSearchCV(clf,param_dist, scoring='roc_auc', n_jobs=2,
                           verbose=1,cv=10)

        # run randomized search
        # n_iter_search = 100
        # clf = RandomizedSearchCV(clf, param_distributions=param_dist,
        #                                    n_iter=n_iter_search, scoring = 'roc_auc',cv=10)

        start = time()
        clf.fit(Train_DS, y)

        # print("RandomizedSearchCV took %.2f seconds for %d candidates"
        #         " parameter settings." % ((time() - start), n_iter_search))
        report(clf.grid_scores_)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        print(clf.grid_scores_)
        print(clf.best_score_)
        print(clf.best_params_)
        print(clf.scorer_)
    else:
        #Starting model
        #clf = SVC(kernel='rbf', class_weight='auto',C=1e5, gamma= 0.01,probability=True)

        #Model with rank: 1 - Grid Search (Cv : 0.815 std: 0.060) - LB : 0.81
        #clf = SVC(kernel='rbf', class_weight='auto',C=10000, gamma= 0.0001,probability=True)

        #Model with rank: 1 - Grid Search (Cv : 0.85 - for new features
        clf = SVC(kernel='rbf', class_weight='auto',C=5000, gamma= 0.0005,probability=True)

        clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set

    preds = pd.DataFrame(pred_Actual, index=Sample_DS.bidder_id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_SVC_306.csv', index_label='bidder_id')

    print("***************Ending SVM with rbf Classifier***************")
    return pred_Actual

########################################################################################################################
#Misc  Classifiers
########################################################################################################################
def Misc_Model_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting Misc Model Classifier***************")
    t0 = time()

    Train_DS = np.log( 1 + Train_DS)
    Actual_DS = np.log( 1 + Actual_DS)

    #Setting Standard scaler for data
    # stdScaler = StandardScaler()
    # stdScaler.fit(Train_DS,y)
    # Train_DS = stdScaler.transform(Train_DS)
    # Actual_DS = stdScaler.transform(Actual_DS)

    if grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        # param_dist = {
        #           'alpha': [0.0001, 0.0005, 0.001, 0.01, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1],
        #           'fit_prior':[True, False]
        #              }
        #
        # clf = MultinomialNB()


        param_dist = {
                  'n_neighbors': [1,2,3,4,5,6,7],
                  'weights':['uniform', 'distance'],
                  'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'leaf_size':[10,20,25,30,35],
                     }
        clf = KNeighborsClassifier()
        clf = GridSearchCV(clf,param_dist, scoring='roc_auc', n_jobs=2,
                           verbose=1,cv=10)

        # run randomized search
        # n_iter_search = 100
        # clf = RandomizedSearchCV(clf, param_distributions=param_dist,
        #                                    n_iter=n_iter_search, scoring = 'roc_auc',cv=10)

        start = time()
        clf.fit(Train_DS, y)

        # print("RandomizedSearchCV took %.2f seconds for %d candidates"
        #         " parameter settings." % ((time() - start), n_iter_search))
        report(clf.grid_scores_)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        print(clf.grid_scores_)
        print(clf.best_score_)
        print(clf.best_params_)
        print(clf.scorer_)
    else:

        #LDA Model - CV : .889
        #clf = LDA(solver='svd')

        #SGD = .63
        #clf=SGDClassifier(loss='log')

        #
        #clf = GMM()

        #Gaussian - CV Score: 0.873520801666
        #clf =GaussianNB()


        #grid search ..CV :0.88
        clf = MultinomialNB()

        #Cv = 0.81
        #clf = BernoulliNB()

        #Best model CV : 0.783
        #clf = KNeighborsClassifier(n_neighbors=7,weights = 'distance',leaf_size=30,algorithm='auto')

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        clf.fit(Train_DS, y)

        # print(clf.class_prior_)
        # print(clf.class_count_)
        # print(clf.theta_)
        # print(clf.sigma_)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set

    preds = pd.DataFrame(pred_Actual, index=Sample_DS.bidder_id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_MM.csv', index_label='bidder_id')

    print("***************Ending Misc Model Classifier***************")
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    global file_path
    file_path = 'C:/Python/Others/data/Kaggle/Facebook_Human_or_Robot/'
    #file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Facebook_Human_or_Robot/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS =  pd.read_csv(file_path+'Train_New6_edited.csv',sep=',')
    Actual_DS =  pd.read_csv(file_path+'Test_New6.csv',sep=',')
    Sample_DS = pd.read_csv(file_path+'sampleSubmission.csv',sep=',')
    Bids_DS = pd.read_csv(file_path+'Bids_New6.csv',sep=',')

    Train_DS, Actual_DS, y =  Data_Munging(Train_DS, Actual_DS, Bids_DS)

    pred_Actual = RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = xgb_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = LogReg_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = SVM_rbf_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = Misc_Model_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
########################################################################################################################
#Get the predictions for actual data set
########################################################################################################################

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)