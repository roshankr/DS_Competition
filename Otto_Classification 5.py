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
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from time import time
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import randint as sp_randint
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit
from random import shuffle

########################################################################################################################
#Otto Classification                                                                                                   #
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
def Data_Munging(Train_DS,Actual_DS):

    print("***************Starting Data cleansing***************")

    #Train_DS['target'] = Train_DS['target'].map(lambda x: x.strip('Class_'))
    y = Train_DS.target.values

    # encode Y (label)
    global label_enc
    label_enc = preprocessing.LabelEncoder()
    y = label_enc.fit_transform(y)

    #Remove not used features
    Train_DS = Train_DS.drop(['id', 'target'], axis=1)
    Actual_DS = Actual_DS.drop(['id'], axis=1)

    #Setting Standard scaler for data
    # stdScaler = StandardScaler()
    # stdScaler.fit(Train_DS,labels)
    # Train_DS = stdScaler.transform(Train_DS)
    # Actual_DS = stdScaler.transform(Actual_DS)

    #apply PCA for train , CV , test and actual DS
    # pca = PCA(n_components=2)
    # pca.fit(Train_DS,labels)
    # Train_DS = pca.transform(Train_DS)
    # Actual_DS = pca.transform(Actual_DS)
    #
    # #To get a better understanding of interaction of the dimensions
    # #plot the three PCA dimensions
    # classes = ['0','1','2','3','4','5','6','7','8']
    # fig = plt.figure(1, figsize=(10, 10))
    # ax = Axes3D(fig, elev=-150, azim=110)
    # ax.scatter(Train_DS[:, 0], Train_DS[:, 1], labels,cmap=plt.cm.Paired,c=labels, label=classes)
    # ax.set_title("First three PCA directions")
    # ax.set_xlabel("1st eigenvector")
    # ax.set_ylabel("2nd eigenvector")
    # ax.set_zlabel("Actual")
    # ax.legend( loc='upper left',numpoints=1,ncol=3,fontsize=8)
    # plt.show()

    # print("***************Ending Data cleansing***************")
    #
    return Train_DS, Actual_DS, y

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Kfold_Cross_Valid(X, y, clf):

    print("***************Starting Kfold Cross validation***************")

    #y = y.astype(str)

    scores=[]
    ss = StratifiedShuffleSplit(y, n_iter=10,test_size=0.2, random_state=42, indices=None)
    #ss=KFold(len(y), n_folds=len(y),shuffle=True,indices=False)

    for trainCV, testCV in ss:
        print(trainCV)
        print(testCV)
        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)

        scores.append(log_loss(y_test,y_pred, eps=1e-15, normalize=True ))

    #Average RMSE from cross validation
    scores=np.array(scores)
    print ("CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def RFC_Classifier(Train_DS, y, Actual_DS, grid=True):
    print("***************Starting Random Forest Classifier***************")
    t0 = time()

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
        clf = RandomForestClassifier(n_estimators=10,n_jobs=1)
        Kfold_score = Kfold_Cross_Valid(Train_DS, y, clf)
        clf.fit(Train_DS, y)

    #incase if it is required for stacking
    pred_Train = clf.predict_proba(Train_DS)

    #Predict actual model
    pred_Actual = clf.predict_proba(Actual_DS)
    print("Actual Model predicted")

    print("***************Ending Random Forest Classifier***************")
    return pred_Train, pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    global file_path
    file_path = 'C:/Python/Others/data/Kaggle/Otto_Product_Classification/'
    #file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Otto_Product_Classification/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS =  pd.read_csv(file_path+'train.csv',sep=',')
    Actual_DS =  pd.read_csv(file_path+'test.csv',sep=',')
    Sample_DS = pd.read_csv(file_path+'sampleSubmission.csv',sep=',')

    Train_DS,Actual_DS,y =  Data_Munging(Train_DS,Actual_DS)

    pred_Train, pred_Actual = RFC_Classifier(Train_DS, y, Actual_DS, grid=False)

########################################################################################################################
#Get the predictions for actual data set
########################################################################################################################
    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'Submission_Roshan.csv', index_label='id')

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)

