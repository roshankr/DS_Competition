import requests
import numpy as np
import scipy as sp
import sys
import math
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
from sklearn.cross_validation import StratifiedShuffleSplit , KFold
from sklearn.calibration import CalibratedClassifierCV
from random import shuffle
import datetime
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
# import xgboost as xgb
# from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
# from lasagne.updates import nesterov_momentum,adagrad
# from lasagne.objectives import binary_crossentropy
# from nolearn.lasagne import NeuralNet
# import theano
# from theano import tensor as T
# from theano.tensor.nnet import sigmoid
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

########################################################################################################################
#West Nile Virus Prediction                                                                                            #
########################################################################################################################

########################################################################################################################
#Computing the distance between two locations on Earth from coordinates
########################################################################################################################
def distance_on_unit_sphere(lat1, long1, lat2, long2):

    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )*3960
    return arc

########################################################################################################################
#
########################################################################################################################

class AdjustVariable(object):
    def __init__(self, variable, target, half_life=4):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2**(1.0/self.half_life)
        self.variable.set_value(np.float32(self.target + delta))
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
def Kfold_Cross_Valid(X,y,clf):

    print("***************Starting Kfold Cross validation***************")

    scores=[]
    ss=KFold(len(y), n_folds=50,shuffle=True,indices=False)
    i = 1
    for trainCV, testCV in ss:

        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        clf.fit(X_train, np.log(y_train))
        y_pred=np.exp(clf.predict(X_test))
        scores.append(mean_squared_error(y_test,y_pred))
        print(" %d-iteration... %s " % (i,scores))
        i = i + 1

    #Average RMSE from cross validation
    scores=np.array(scores)
    print ("CV Score:",np.mean(scores**0.5))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS):

    print("***************Starting Data cleansing***************")

    #Train_DS = np.array(Train_DS)
    y = Train_DS['53']
    Train_DS = Train_DS.drop(['53'], axis = 1)
    #Actual_DS = Actual_DS.drop(['52'], axis = 1)

    #y = Train_DS.WnvPresent.values
    #Train_DS = Train_DS.drop(['id','WnvPresent'], axis = 1)
    #Actual_DS = Actual_DS.drop(['id'], axis = 1)

    #Do we need to work on spray data ??????
    #row_iterator = Train_DS.iterrows()
    #test = distance_on_unit_sphere(41.750498,-87.605294, 42.3916233333333, -88.0891633333333)

    # Train_DS_next = Train_DS.groupby(['year','month','Trap','Species']).agg({'NumMosquitos': [np.mean]}).reset_index()
    # Train_DS_next['year'] = Train_DS_next['year'] + 1
    # Train_DS_next.columns = ['year','month','Trap','Species','NumMosquitos1']
    # Actual_DS = Actual_DS.merge(Train_DS_next, on=['year','month','Trap','Species'],how='left')
    #
    # Train_DS_next = Train_DS.groupby(['year','Species']).agg({'NumMosquitos': [np.mean]}).reset_index()
    # Train_DS_next['year'] = Train_DS_next['year'] + 1
    # Train_DS_next.columns = ['year','Species','NumMosquitos2']
    # Actual_DS = Actual_DS.merge(Train_DS_next, on=['year','Species'],how='left')
    #
    # print(Train_DS.columns)
    #
    #
    # #print(Train_DS_next)
    # Actual_DS.to_csv(file_path+'Actual_DS_group.csv')
    # sys.exit(0)
    ###################################################################################################################
    #Prediction for Nummosquitos
    # cols = ['year','month','day','dayofweek','Trap','Species','NumMosquitos']
    # Train_DS_Mosq  = Train_DS[cols]
    # Actual_DS_Mosq = Actual_DS[cols]
    # Train_DS_Mosq_y = Train_DS_Mosq['NumMosquitos']
    # Train_DS_Mosq  = Train_DS_Mosq.drop(['NumMosquitos'], axis = 1)
    # Actual_DS_Mosq = Actual_DS_Mosq.drop(['NumMosquitos'], axis = 1)
    #
    # clf = RandomForestRegressor(n_estimators=100)
    #
    # scores = Kfold_Cross_Valid(Train_DS_Mosq,Train_DS_Mosq_y,clf)
    #
    # clf.fit(Train_DS_Mosq,np.log(Train_DS_Mosq_y))
    #
    # Y_pred=np.exp(clf.predict(Train_DS_Mosq))
    #
    # #Average RMSE from train
    # scores = (mean_squared_error(Train_DS_Mosq_y,Y_pred))**0.5
    # print ("CV Score:",scores)
    #
    # Train_DS['NumMosquitos'] = Y_pred
    #
    # Y_pred=np.exp(clf.predict(Actual_DS_Mosq))
    # Actual_DS['NumMosquitos'] = Y_pred

    #Actual_DS.to_csv(file_path+'Actual_DS_group.csv')
    ###################################################################################################################

    print("***************Ending Data cleansing***************")

    return Train_DS, Actual_DS, y

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid(X, y, clf):

    print("***************Starting Kfold Cross validation***************")
    X =np.array(X)
    scores=[]
    ss = StratifiedShuffleSplit(y, n_iter=10,test_size=0.3, random_state=42, indices=None)
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
    print ("Normal CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid_multi_iter(X, y, clf):

    print("***************Starting Kfold Cross validation***************")

    print(np.shape(X))

    scores=[]

    X['y'] = y
    year_vale = [2007,2009,2011,2013]

    for i in year_vale:
        X_train = X[X['year'] != i]
        X_test  = X[X['year'] == i]
        y_train = X_train['y']
        y_test  = X_test['y']
        X_train = X_train.drop(['y'], axis=1)
        X_test    = X_test.drop(['y'], axis=1)

        X_train = np.array(X_train)
        X_test  = np.array(X_test)
        y_train = np.array(y_train)
        y_test  = np.array(y_test)

        ss = StratifiedShuffleSplit(y_test, n_iter=10,test_size=0.2, random_state=42, indices=None)

        scores1 =[]
        iter = 1
        for trainCV, testCV in ss:

            X_train1, X_test1 = X_train[trainCV], X_test[testCV]
            y_train1, y_test1 = y_train[trainCV], y_test[testCV]

            # print(np.shape(X_train1))
            # print(np.shape(X_test1))
            # print(np.shape(y_train1))
            # print(np.shape(y_test1))
            # print(y_test1)

            clf.fit(X_train1, y_train1)

            y_pred=clf.predict_proba(X_test1)[:,1]
            scores1.append(roc_auc_score(y_test1, y_pred))
            print(" %d-iteration... %s " % (iter,scores1))
            iter = iter + 1

        scores1=np.array(scores1)
        scores.append(np.mean(scores1))
        print ("4 fold (year) CV Score:",np.mean(scores1))


    #Average ROC from cross validation
    scores=np.array(scores)
    print ("4 fold (year) CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid_new(X, y, clf):

    print("***************Starting Kfold Cross validation***************")

    scores=[]
    iter = 1
    X['preds'] = y
    year_vale = [2007,2009,2011,2013]

    for i in year_vale:
        X_train = X[X['year'] != i]
        X_test  = X[X['year'] == i]
        y_train = X_train['preds']
        y_test  = X_test['preds']
        X_train = X_train.drop(['preds'], axis=1)
        X_test  = X_test.drop(['preds'], axis=1)

        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        clf.fit(X_train, y_train)

        y_pred=clf.predict_proba(X_test)[:,1]
        scores.append(roc_auc_score(y_test, y_pred))
        print(" %d-iteration... %s " % (iter,scores))
        iter = iter + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("4 fold (year) CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):
    print("***************Starting Random Forest Classifier***************")
    t0 = time()

    if grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "max_depth": [1, 2, 3, 4, 5, None],
                      "max_features": sp_randint(1, 40),
                      "min_samples_split": sp_randint(1, 20),
                      "min_samples_leaf": sp_randint(1, 20),
                      "bootstrap": [True, False]
                     }

        clf = RandomForestClassifier(n_estimators=100,n_jobs=-1)

        # run randomized search
        n_iter_search = 100
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = 'roc_auc',cv=10)

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

        #starting model (benchmark) - .55 in CV, .67 in LB
        clf = RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=1)

        # clf = RandomForestClassifier(min_samples_leaf=12,bootstrap=False,n_jobs=1, n_estimators=1000,
        #                               min_samples_split=14,max_depth=1,max_features=5)

        #Best from Random Search
        # clf = RandomForestClassifier(min_samples_leaf=17,bootstrap=True,n_jobs=-1, n_estimators=100,
        #                               min_samples_split=3,max_depth=1,max_features=31)

        scores = Nfold_Cross_Valid(Train_DS, y, clf)

        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_RF.csv', index_label='id')

    print("***************Ending Random Forest Classifier***************")
    return pred_Actual

########################################################################################################################
#XGB Classifier
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

        #starting model - Cv:0.82
        clf = xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,nthread=2,min_child_weight=1,
                             subsample=0.9,colsample_bytree=1,silent=True, gamma = 0.7)


        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set

    preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_xgb.csv', index_label='id')

    print("***************Ending XGB Classifier***************")
    return pred_Actual

########################################################################################################################
#SVM with rbf  Classifier
########################################################################################################################
def SVM_rbf_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting SVM with rbf Classifier***************")

    t0 = time()

    #Setting Standard scaler for data
    # stdScaler = StandardScaler()
    # stdScaler.fit(Train_DS,y)
    # Train_DS = stdScaler.transform(Train_DS)
    # Actual_DS = stdScaler.transform(Actual_DS)

    #Train_DS = np.log( 1 + Train_DS)
    #Actual_DS = np.log( 1 + Actual_DS)

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

        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        scores = Nfold_Cross_Valid(Train_DS, y, clf)

        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,1]
    print("Actual Model predicted")

    #Get the predictions for actual data set

    preds = pd.DataFrame(pred_Actual, index=Sample_DS.bidder_id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_SVC_305.csv', index_label='bidder_id')

    print("***************Ending SVM with rbf Classifier***************")
    return pred_Actual

#########################################################################################################################
#Neural Network Classifier 1
########################################################################################################################
def NN1_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting NN1 Classifier***************")
    t0 = time()

    if grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

    else:
        y = y.reshape((-1, 1))
        Train_DS = np.log( 1 + Train_DS)
        Actual_DS = np.log( 1 + Actual_DS)
        #
        #Setting Standard scaler for data
        # stdScaler = StandardScaler()
        # stdScaler.fit(Actual_DS)
        # Train_DS = stdScaler.transform(Train_DS)
        # Actual_DS = stdScaler.transform(Actual_DS)


        learning_rate = theano.shared(np.float32(0.1))

        # y = y.astype('float32')
        # Train_DS = Train_DS.astype('float32')
        # Actual_DS = Actual_DS.astype('float32')

        #Define Model parms - 2 hidden layers
        clf = NeuralNet(
              layers=[
                ('input', InputLayer),
                ('hidden1', DenseLayer),
                ('dropout1', DropoutLayer),
                ('hidden2', DenseLayer),
                ('dropout2', DropoutLayer),
                ('output', DenseLayer),
                    ],

   	    # layer parameters:
        input_shape=(None, Train_DS.shape[1]),
        hidden1_num_units=400,
        dropout1_p=0.4,
        hidden2_num_units=400,
        dropout2_p=0.4,
        output_nonlinearity=sigmoid,
        output_num_units=1,

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=learning_rate,
        update_momentum=0.9,

        # Decay the learning rate
        on_epoch_finished=[
                AdjustVariable(learning_rate, target=0, half_life=2),
                ],

        # This is silly, but we don't want a stratified K-Fold here
        # To compensate we need to pass in the y_tensor_type and the loss.
        regression=True,
        y_tensor_type = T.imatrix,
        objective_loss_function = binary_crossentropy,

        max_epochs=85,
        eval_size=0.1,
        verbose=1,
        )

        Train_DS, y = shuffle(Train_DS, y, random_state=42)
        clf.fit(Train_DS, y)

        _, X_valid, _, y_valid = clf.train_test_split(Train_DS, y, clf.eval_size)
        probas = clf.predict_proba(X_valid)[:,0]
        print("ROC score", metrics.roc_auc_score(y_valid, probas))

        print("done in %0.3fs" % (time() - t0))

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)[:,0]
    print("Actual NN1 Model predicted")

    #Get the predictions for actual data set
    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_NN1.csv', index_label='id')

    print("***************Ending NN1 Classifier***************")
    return pred_Actual


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
    Train_DS =  pd.read_csv(file_path+'Train_DS_Lasagne.csv',sep=',')
    Actual_DS =  pd.read_csv(file_path+'Actual_DS_Lasagne.csv',sep=',')
    Sample_DS = pd.read_csv(file_path+'sampleSubmission.csv',sep=',')
    Weather_DS = pd.read_csv(file_path+'weather4.csv',sep=',')

    Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS)

    pred_Actual = RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = xgb_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = SVM_rbf_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = NN1_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
########################################################################################################################
#Get the predictions for actual data set
########################################################################################################################


########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)

