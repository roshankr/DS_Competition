import requests
import numpy as np
import random
import scipy as sp
import sys
from random import sample ,shuffle
import pandas as pd # pandas
from sklearn.svm import SVR
from time import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LogisticRegression, \
    Perceptron,RidgeCV
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from random import shuffle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures,Imputer
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA,KernelPCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from scipy.linalg import solve
from scipy.sparse.linalg import lsqr
#sys.path.append('C:/Python34/Lib/site-packages/xgboost')
#import xgboost as xgb

########################################################################################################################
#Restaurant Revenue Prediction using PCA,recursive feature extraction                                                  #
########################################################################################################################

########################################################################################################################
#Data reading and cleaning
########################################################################################################################
def Data_Munging(train,test,Sample_DS):

    print("***************Starting Data clean up***************")

    #calculate the age of each sample.
    train["Age"] = 2015 - pd.DatetimeIndex(train['Open Date']).year
    test["Age"] = 2015 - pd.DatetimeIndex(test['Open Date']).year

    train["days"] = ((pd.DatetimeIndex(train['Open Date']).year)*365) + (pd.DatetimeIndex(train['Open Date']).dayofyear)
    test["days"]  = ((pd.DatetimeIndex(test['Open Date']).year)*365)  + (pd.DatetimeIndex(test['Open Date']).dayofyear)

    cols = train.columns[2:42]
    X_New = train[cols]
    Xt_New = test[cols]

    #Extract the age and log transform it
    X=np.log(train[['Age']].values.reshape((train.shape[0],1)))
    Xt=np.log(test[['Age']].values.reshape((test.shape[0],1)))
    y=train['revenue'].values

    #Consolidate Types
    X_New['Type']  = np.where (X_New['Type'] == 'DT', 'IL', X_New['Type'])
    X_New['Type']  = np.where (X_New['Type'] == 'MB', 'FC', X_New['Type'])
    Xt_New['Type'] = np.where (Xt_New['Type'] == 'DT', 'IL', Xt_New['Type'])
    Xt_New['Type'] = np.where (Xt_New['Type'] == 'MB', 'FC', Xt_New['Type'])

    # Use all values in City , City Group and Type as a different column for Train and Actual DS
    for column in ['City','City Group','Type']:
        dummies = pd.get_dummies(X_New[column])
        X_New[dummies.columns] = dummies

    for column in ['City','City Group','Type']:
        dummies = pd.get_dummies(Xt_New[column])
        Xt_New[dummies.columns] = dummies

    # Take the remaining columns and add to the X_New or Xt_New
    list2 = list(set(X_New.columns) - set(Xt_New.columns))
    list1 = list(set(Xt_New.columns) - set(X_New.columns))

    df1 = pd.DataFrame(np.zeros(shape=(len(X_New),len(list1)), dtype=int),columns=list1)
    X_New = pd.concat([X_New,df1],axis=1)

    df1 = pd.DataFrame(np.zeros(shape=(len(Xt_New),len(list2)), dtype=int),columns=list2)
    Xt_New = pd.concat([Xt_New,df1],axis=1)

    #global label_enc
    # label_City = preprocessing.LabelEncoder()
    # label_City.fit(pd.concat([train['City'],test['City']],axis=0))
    # X_New['City'] = np.log(1+ label_City.transform(train['City']))
    # Xt_New['City'] = np.log(1+ label_City.transform(test['City']))
    #
    # label_CityG = preprocessing.LabelEncoder()
    # label_CityG.fit(pd.concat([train['City Group'],test['City Group']],axis=0))
    # X_New['City Group'] = np.log(1+ label_CityG.transform(train['City Group']))
    # Xt_New['City Group'] = np.log(1+ label_CityG.transform(test['City Group']))
    #
    # label_Type = preprocessing.LabelEncoder()
    # label_Type.fit(pd.concat([train['Type'],test['Type']],axis=0))
    # X_New['Type'] = np.log(1+ label_Type.transform(train['Type']))
    # Xt_New['Type'] = np.log(1+ label_Type.transform(test['Type']))

    #log transform all P variables
    cols = X_New.columns[3:40]

    #use imputer for missing values
    imp = Imputer(missing_values=0, strategy='mean', axis=0,copy=False)
    X_New[cols] = imp.fit_transform(X_New[cols])
    Xt_New[cols] = imp.fit_transform(Xt_New[cols])

    X_New[cols] = np.log(1+X_New[cols])
    Xt_New[cols] = np.log(1+Xt_New[cols])

    #Move Age values to Train and actual DS
    X_New["Age"] = X
    Xt_New["Age"] = Xt

    #Keep the Train and Actual feature DS in X and Xt
    X = X_New
    Xt = Xt_New

    X['days'] = np.log(1+train['days'])
    Xt['days'] = np.log(1+test['days'])
    print("***************Ending Data clean up***************")

    return X,Xt,y

########################################################################################################################
#Feature selection and extraction
########################################################################################################################
def Feature_Selection(X,Xt,y):

    print("***************Starting Feature Selection***************")

    X = X.drop(['Type','City Group','City'],axis=1)
    Xt = Xt.drop(['Type','City Group','City'],axis=1)

    #reorder the datasets in column order so that both Train and Actual DS looks similar
    X = X.reindex_axis(sorted(X.columns), axis=1)
    Xt = Xt.reindex_axis(sorted(Xt.columns), axis=1)

    ##################################Use PCA for feature extraction####################################################
    pca = PCA(30)
    #pca = KernelPCA(n_components=20,kernel='rbf')
    pca.fit(Xt)
    X = pca.transform(X)
    Xt = pca.transform(Xt)
    #print(pca.explained_variance_ratio_ )

    #To get a better understanding of interaction of the dimensions
    #plot the three PCA dimensions

    # fig = plt.figure(2, figsize=(8, 6))
    # ax = Axes3D(fig, elev=-150, azim=110)
    #
    # #ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2],facecolors='none', edgecolors='r')
    # ax.scatter(X[:, 0], X[:, 1], np.log(y),color='g',marker='*')
    # # ax.set_xlim([70, -20])
    # # ax.set_ylim([-1.4, 1])
    # # ax.set_zlim([1, -1])
    # ax.set_title("First three PCA directions")
    # ax.set_xlabel("1st eigenvector")
    # ax.set_ylabel("2nd eigenvector")
    # ax.set_zlabel("Actual")
    # plt.show()

    print("***************Ending Feature Selection***************")
    return X,Xt,y

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Kfold_Cross_Valid(X,Xt,y,clf):

    print("***************Starting Kfold Cross validation***************")

    #clf = ElasticNet(alpha=0.1,l1_ratio=0.3,max_iter=10000)
    scores=[]
    ss=KFold(len(y), n_folds=len(y),shuffle=True,indices=False)

    for trainCV, testCV in ss:

        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        clf.fit(X_train, np.log(y_train))
        y_pred=np.exp(clf.predict(X_test))

        scores.append(mean_squared_error(y_test,y_pred))

    #Average RMSE from cross validation
    scores=np.array(scores)
    print ("CV Score:",np.mean(scores**0.5))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#grid search for Cv and parameter set up and return the best model
########################################################################################################################
def GridSrch_Modelfit(X,Xt,y,grid):

    print("***************Starting Grid Search and model fit***************")


    #Split into Training and Test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scores=[]

    if grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        ###########################parm for SVR#########################################################################
        param_grid = {'kernel' : ['rbf','poly','sigmoid'], 'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.01, 0.1], }
        clf = GridSearchCV(SVR(),param_grid, scoring='mean_squared_error', n_jobs=3,
                          verbose=1)

        ####################################parm for RF#################################################################
        # param_grid = {'n_estimators': [100], 'max_depth': [None, 1, 2, 3, 4,  5],
        #              'min_samples_split': [1, 3, 5],'max_features':['auto','sqrt','log2',None] }
        # clf = GridSearchCV(RandomForestRegressor(),param_grid, scoring='mean_squared_error', n_jobs=2,
        #                   verbose=1,cv=10)

        # RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
        #    max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
        #    min_samples_split=1, min_weight_fraction_leaf=0.0,
        #    n_estimators=1000, n_jobs=1, oob_score=False, random_state=None,
        #    verbose=0, warm_start=False)

        ####################################parm for Elasticnet#########################################################
        # param_grid = {'alpha': [1, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.01 , 0.001] ,
        #               'l1_ratio': [.2, .1, .3, .4,.5, .6,.7, .8,.9],
        #              'selection': ['cyclic', 'random'],'fit_intercept':[True,False],'normalize':[True,False],
        #               'positive':[True,False]}
        #
        # clf = GridSearchCV(ElasticNet(max_iter=25000),param_grid, scoring='mean_squared_error', n_jobs=3,
        #                   verbose=1)
        ####################################parm for Ridge##############################################################
        # param_grid = {'alpha': [100, 10, 0.1, 0.01 , 0.001] , 'fit_intercept':[True,False],'normalize':[True,False],
        #              'selection': ['auto', 'svd','sparse_cg','cholesky','lsqr']}
        #
        # clf = GridSearchCV(Ridge(max_iter=10000),param_grid,scoring='mean_squared_error',  n_jobs=3,
        #                   verbose=1,cv=20)
        ####################################parm for Lasso##############################################################
        # param_grid = {'alpha': [10, 0.5, 0.1, 0.01 , 0.001] , 'fit_intercept':[True,False],'normalize':[True,False],
        #              'selection': ['cyclic', 'random']}
        #
        # clf = GridSearchCV(Lasso(max_iter=10000),param_grid,scoring='mean_squared_error',  n_jobs=3,
        #                   verbose=1,cv=20)
        ################################################################################################################
        # param_grid = {
        # 	          'loss': [ 'squared_loss', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive'] ,
        # 	          'penalty': [ None, 'l1', 'l2', 'elasticnet'] ,
        # 	          'alpha': [1, 0.5, 0.001,0.0001] ,
        # 	          'l1_ratio': [.2, .1, .3, .4,.5, .6,.7, .8,.9],
        # 	          'fit_intercept':[True,False],
        # 	          'shuffle':[True,False],
        # 	          'verbose':[0,1],
        # 	          'epsilon': [.1, .01, .001, .0001],
        # 	          'learning_rate': ['constant','optimal','invscaling'],
        # 	          'power_t':[1 , .25, .01, .1],
        # 	          'average': [True,False, 10,20,30]
        #              }
        #
        # clf = GridSearchCV(SGDRegressor(),param_grid, scoring='mean_squared_error', n_jobs=3,
        #                   verbose=1,cv=10)
        ################################################################################################################
        #clf.fit(X_train,np.log(Y_train))
        clf.fit(X,np.log(y))

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        print(clf.grid_scores_)
        print(clf.best_score_)
        print(clf.best_params_)
        print(clf.scorer_)

    else:
        print("Starting model fit without Grid Search")

        #clf = ElasticNet(alpha=0.1,l1_ratio=0.3,max_iter=10000)

        #CV Score: 1533507.8203 , LB = 1765769.69574, best as of now ******************************************
        # clf = ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.1,
        #         max_iter=10000, normalize=False, positive=False, precompute=False,
        #         random_state=None, selection='random', tol=0.0001, warm_start=False)

        # clf = SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.01,
        #      fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
        #      loss='squared_loss', n_iter=5, penalty='l2', power_t=0.25,
        #      random_state=None, shuffle=True, verbose=0, warm_start=False)

        #CV Score: 1549297.12445 , LB = 1769900.29113
        # clf = ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.2,
        # max_iter=25000, normalize=False, positive=False, precompute=False,
        # random_state=None, selection='cyclic', tol=0.0001, warm_start=False)

        # clf=Ridge(alpha=1, copy_X=True, fit_intercept=True, max_iter=10000,
        #     normalize=False, solver='auto', tol=0.001)

        #clf = Lasso(alpha = 0.1)
        #
        # clf = SVR(C=1000.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.001, gamma=0.1,
        #        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

        # clf = SVR(C=1000.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.01, gamma=0.0001,
        #     kernel='sigmoid', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

        #current one
        #clf = RandomForestRegressor(n_estimators=1500,max_features=None,min_samples_split=1 ,max_depth=None,n_jobs=2)

        clf =  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=1, min_weight_fraction_leaf=0.0,
           n_estimators=1000, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

        #clf = GradientBoostingRegressor(n_estimators=1500 , learning_rate = 0.001, max_features = None,min_samples_split=1
        #                                ,max_depth=None)
        #clf = SVR(kernel='rbf', C=1e3, gamma=0.01)


        #clf = KNeighborsRegressor(n_neighbors=10,algorithm = '')

        #Try Adaboost on Decision tree
        #clf = AdaBoostRegressor(RandomForestRegressor(n_estimators=1000,n_jobs=1,max_features="sqrt"),n_estimators=300,
        #                                                                                      loss='square')

        Kfold_score = Kfold_Cross_Valid(X,Xt,y,clf)


        clf.fit(X,np.log(y))
        #print(clf.feature_importances_)

    Y_pred=np.exp(clf.predict(X_test))

    #Average RMSE from cross validation
    scores = (mean_squared_error(Y_test,Y_pred))**0.5
    print ("CV Score:",scores)

    print("***************Ending Grid Search and model fit***************")

    return clf

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################

def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)
    pd.options.mode.chained_assignment = None  # default='warn'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    train =  pd.read_csv('C:/Python/Others/data/Kaggle/Restaurant_Revenue_Prediction/train.csv',sep=',')
    test =  pd.read_csv('C:/Python/Others/data/Kaggle/Restaurant_Revenue_Prediction/test.csv',sep=',')
    Sample_DS = pd.read_csv('C:/Python/Others/data/Kaggle/Restaurant_Revenue_Prediction/sampleSubmission.csv',sep=',')

    X,Xt,y = Data_Munging(train,test,Sample_DS)

    X,Xt,y = Feature_Selection(X,Xt,y)

    #scores = Kfold_Cross_Valid(X,Xt,y)

    clf = GridSrch_Modelfit(X,Xt,y,grid=False)

    #Predict test.csv & reverse the log transform
    yp=np.exp(clf.predict(Xt))

########################################################################################################################
#Get the predictions for actual data set
########################################################################################################################
    #Get the predictions for actual data set
    preds = pd.DataFrame(yp, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    preds.to_csv('C:/Python/Others/data/Kaggle/Restaurant_Revenue_Prediction/Submission_Roshan.csv', index_label='Id')

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)