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
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

########################################################################################################################
#Otto Classification                                                                                                   #
########################################################################################################################

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def RFC_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test, Actual_DS):
    print("***************Starting Random Forest Classifier***************")
    t0 = time()
    clf = RandomForestClassifier(n_estimators=500,n_jobs=1)
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_cv)
    score = clf.score(X_cv,Y_cv)

    print("Random Forest Classifier - {0:.2f}%".format(100 * score))
    Summary = pd.crosstab(label_enc.inverse_transform(Y_cv), label_enc.inverse_transform(preds),
                      rownames=['actual'], colnames=['preds'])
    Summary['pct'] = (Summary.divide(Summary.sum(axis=1), axis=1)).max(axis=1)*100
    print(Summary)

    #Check with log loss function
    epsilon = 1e-15
    #ll_output = log_loss_func(Y_cv, preds, epsilon)
    preds2 = clf.predict_proba(X_cv)
    ll_output2= log_loss(Y_cv, preds2, eps=1e-15, normalize=True)
    print(ll_output2)
    print("done in %0.3fs" % (time() - t0))

    preds3 = clf.predict_proba(X_test)

    print("x_test done")
    #preds4 = clf.predict_proba((Actual_DS.ix[:,'feat_1':]))
    preds4 = clf.predict_proba(Actual_DS)

    print("***************Ending Random Forest Classifier***************")
    return pd.DataFrame(preds2) , pd.DataFrame(preds3),pd.DataFrame(preds4)

########################################################################################################################
#SVM
########################################################################################################################
def SGDC_SVM_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test, Actual_DS):
    print("***************Starting SVM***************")
    t0 = time()
    clf = SGDClassifier(loss='log', penalty='l2',alpha=1e-5, n_iter=100)
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_cv)
    score = clf.score(X_cv,Y_cv)

    print("{0:.2f}%".format(100 * score))
    Summary = pd.crosstab(label_enc.inverse_transform(Y_cv), label_enc.inverse_transform(preds),
                      rownames=['actual'], colnames=['preds'])
    Summary['pct'] = (Summary.divide(Summary.sum(axis=1), axis=1)).max(axis=1)*100
    print(Summary)

    #Check with log loss function
    epsilon = 1e-15
    #ll_output = log_loss_func(Y_cv, preds, epsilon)
    preds2 = clf.predict_proba(X_cv)
    ll_output2= log_loss(Y_cv, preds2, eps=1e-15, normalize=True)
    print(ll_output2)

    print("done in %0.3fs" % (time() - t0))

    preds3 = clf.predict_proba(X_test)
    #preds4 = clf.predict_proba((Actual_DS.ix[:,'feat_1':]))
    preds4 = clf.predict_proba(Actual_DS)
    print("***************Ending SVM***************")
    return pd.DataFrame(preds2),pd.DataFrame(preds3),pd.DataFrame(preds4)

########################################################################################################################
#SVM with rbf
########################################################################################################################
def SVM_rbf_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test, Actual_DS):
    print("***************Starting SVM with rbf***************")
    t0 = time()
    #used for checking the best performance with 'C' and 'gamma'
    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #           'gamma': [0.0001, 0.0005, 0.001, 0.01, 0.1],}
    # clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto', probability=True), param_grid,
    #                    scoring='log_loss', n_jobs=3, verbose=1,cv=20)

    clf = SVC(kernel='rbf', class_weight='auto',C=1e5, gamma= 0.01,probability=True)
    clf = clf.fit(X_train, Y_train)
    # print("Best estimator found by grid search:")
    # print(clf.best_estimator_)
    preds = clf.predict(X_cv)
    score = clf.score(X_cv,Y_cv)

    print("SVM with rbf - {0:.2f}%".format(100 * score))
    Summary = pd.crosstab(label_enc.inverse_transform(Y_cv), label_enc.inverse_transform(preds),
                      rownames=['actual'], colnames=['preds'])
    Summary['pct'] = (Summary.divide(Summary.sum(axis=1), axis=1)).max(axis=1)*100
    print(Summary)
    #Check with log loss function
    epsilon = 1e-15
    #ll_output = log_loss_func(Y_cv, preds, epsilon)
    preds2 = clf.predict_proba(X_cv)
    ll_output2= log_loss(Y_cv, preds2, eps=1e-15, normalize=True)
    print(ll_output2)

    print("done in %0.3fs" % (time() - t0))

    preds3 = clf.predict_proba(X_test)
    #preds4 = clf.predict_proba((Actual_DS.ix[:,'feat_1':]))
    preds4 = clf.predict_proba(Actual_DS)

    print("***************Ending SVM with rbf***************")
    return pd.DataFrame(preds2),pd.DataFrame(preds3),pd.DataFrame(preds4)

########################################################################################################################
#Gradient Boosting
########################################################################################################################
def GB_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test, Actual_DS):
    print("***************Starting Gradient Boosting***************")
    t0 = time()
    clf = GradientBoostingClassifier(n_estimators=500,learning_rate=0.01)
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_cv)
    score = clf.score(X_cv,Y_cv)

    print("Gradient Boosting - {0:.2f}%".format(100 * score))
    Summary = pd.crosstab(label_enc.inverse_transform(Y_cv), label_enc.inverse_transform(preds),
                      rownames=['actual'], colnames=['preds'])
    Summary['pct'] = (Summary.divide(Summary.sum(axis=1), axis=1)).max(axis=1)*100
    print(Summary)

    #Check with log loss function
    epsilon = 1e-15
    #ll_output = log_loss_func(Y_cv, preds, epsilon)
    preds2 = clf.predict_proba(X_cv)
    ll_output2= log_loss(Y_cv, preds2, eps=1e-15, normalize=True)
    print(ll_output2)

    print("done in %0.3fs" % (time() - t0))

    preds3 = clf.predict_proba(X_test)
    #preds4 = clf.predict_proba((Actual_DS.ix[:,'feat_1':]))
    preds4 = clf.predict_proba(Actual_DS)

    print("***************Ending Gradient Boosting***************")
    return pd.DataFrame(preds2),pd.DataFrame(preds3),pd.DataFrame(preds4)

########################################################################################################################
#Extreme Random Forest Classifier (around 80%)
########################################################################################################################
def ERFC_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test, Actual_DS):
    print("***************Starting Extreme Random Forest Classifier***************")
    t0 = time()
    clf = ExtraTreesClassifier(n_estimators=100,n_jobs=-1)
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_cv)
    score = clf.score(X_cv,Y_cv)

    print("Extreme Random Forest Classifier - {0:.2f}%".format(100 * score))
    Summary = pd.crosstab(label_enc.inverse_transform(Y_cv), label_enc.inverse_transform(preds),
                      rownames=['actual'], colnames=['preds'])
    Summary['pct'] = (Summary.divide(Summary.sum(axis=1), axis=1)).max(axis=1)*100
    print(Summary)

    #Check with log loss function
    epsilon = 1e-15
    #ll_output = log_loss_func(Y_cv, preds, epsilon)
    preds2 = clf.predict_proba(X_cv)
    ll_output2= log_loss(Y_cv, preds2, eps=1e-15, normalize=True)
    print(ll_output2)
    print("done in %0.3fs" % (time() - t0))

    preds3 = clf.predict_proba(X_test)
    #preds4 = clf.predict_proba((Actual_DS.ix[:,'feat_1':]))
    preds4 = clf.predict_proba(Actual_DS)

    print("***************Ending Extreme Random Forest Classifier***************")
    return pd.DataFrame(preds2) , pd.DataFrame(preds3),pd.DataFrame(preds4)

########################################################################################################################
# AdaBoost classifier  Classifier
########################################################################################################################
def ADA_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test, Actual_DS):
    print("***************Starting  AdaBoost Classifier***************")
    t0 = time()
    clf = AdaBoostClassifier(n_estimators=300)
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_cv)
    score = clf.score(X_cv,Y_cv)

    print("AdaBoost Classifier - {0:.2f}%".format(100 * score))
    Summary = pd.crosstab(label_enc.inverse_transform(Y_cv), label_enc.inverse_transform(preds),
                      rownames=['actual'], colnames=['preds'])
    Summary['pct'] = (Summary.divide(Summary.sum(axis=1), axis=1)).max(axis=1)*100
    print(Summary)

    #Check with log loss function
    epsilon = 1e-15
    #ll_output = log_loss_func(Y_cv, preds, epsilon)
    preds2 = clf.predict_proba(X_cv)
    ll_output2= log_loss(Y_cv, preds2, eps=1e-15, normalize=True)
    print(ll_output2)
    print("done in %0.3fs" % (time() - t0))

    preds3 = clf.predict_proba(X_test)
    #preds4 = clf.predict_proba((Actual_DS.ix[:,'feat_1':]))
    preds4 = clf.predict_proba(Actual_DS)

    print("***************Ending AdaBoost Classifier***************")
    return pd.DataFrame(preds2) , pd.DataFrame(preds3),pd.DataFrame(preds4)

########################################################################################################################
#Logistics regression
########################################################################################################################
def LR_Classifier(X_train, X_cv, Y_train, Y_cv, Stack_X_Actual,Sample_DS):
    print("***************Starting LR_Classifier***************")
    t0 = time()
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_cv)
    score = clf.score(X_cv,Y_cv)

    print("LR_Classifier - {0:.2f}%".format(100 * score))
    Summary = pd.crosstab(label_enc.inverse_transform(Y_cv), label_enc.inverse_transform(preds),
                      rownames=['actual'], colnames=['preds'])
    Summary['pct'] = (Summary.divide(Summary.sum(axis=1), axis=1)).max(axis=1)*100
    print(Summary)

    #Check with log loss function
    epsilon = 1e-15
    #ll_output = log_loss_func(Y_cv, preds, epsilon)
    preds2 = clf.predict_proba(X_cv)
    ll_output2= log_loss(Y_cv, preds2, eps=1e-15, normalize=True)
    print(ll_output2)

    #Get the predictions for actual data set
    preds3 = clf.predict_proba(Stack_X_Actual)
    preds3 = pd.DataFrame(preds3, index=Sample_DS.id.values, columns=Sample_DS.columns[1:])
    preds3.to_csv('C:/Python/Others/data/Kaggle/Otto_Product_Classification/Submission_Roshan.csv', index_label='id')

    print("done in %0.3fs" % (time() - t0))
    print("***************Ending LR_Classifier***************")
    return pd.DataFrame(preds2) ,pd.DataFrame(preds3)

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################

def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS =  pd.read_csv('C:/Python/Others/data/Kaggle/Otto_Product_Classification/train.csv',sep=',')
    Actual_DS =  pd.read_csv('C:/Python/Others/data/Kaggle/Otto_Product_Classification/test.csv',sep=',')
    Sample_DS = pd.read_csv('C:/Python/Others/data/Kaggle/Otto_Product_Classification/sampleSubmission.csv',sep=',')

    #Train_DS['target'] = Train_DS['target'].map(lambda x: x.strip('Class_'))
    Y = Train_DS.target.values
    Train_DS = Train_DS.drop(['id', 'target'], axis=1)
    Actual_DS = Actual_DS.drop(['id'], axis=1)

    # encode Y (label)
    global label_enc
    label_enc = preprocessing.LabelEncoder()
    labels = label_enc.fit_transform(Y)

    #Setting Standard scaler for data
    stdScaler = StandardScaler()
    stdScaler.fit(Train_DS,labels)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

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


    #Split into Training and Test sets
    X_train, X_rem, Y_train, Y_rem = train_test_split(Train_DS, labels, test_size=0.4, random_state=42)
    X_cv, X_test, Y_cv, Y_test = train_test_split(X_rem, Y_rem, test_size=0.5, random_state=42)

########################################################################################################################
#Call classifiers
########################################################################################################################
    #p_cv_RFC, p_test_RFC, p_act_RFC = RFC_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test,Actual_DS)
    #p_cv_ERFC, p_test_ERFC, p_act_ERFC = ERFC_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test,Actual_DS)
    #p_cv_SVM, p_test_SVM, p_act_SVM = SGDC_SVM_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test,Actual_DS)
    p_cv_RBF, p_test_RBF , p_act_RBF = SVM_rbf_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test,Actual_DS)
    sys.exit(0)
    p_cv_GBC, p_test_GBC, p_act_GBC = GB_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test,Actual_DS)
    p_cv_ADA, p_test_ADA, p_act_ADA = ADA_Classifier(X_train, X_cv, X_test, Y_train,Y_cv,Y_test,Actual_DS)


    #Taking avg while stacking
    #Stack_X_cv = (p_cv_RFC + p_cv_RBF+p_cv_GBC) / 3
    #Stack_X_test = (p_test_RFC + p_test_RBF+ p_test_GBC ) / 3
    #Stack_X_Actual = (p_act_RFC + p_act_RBF +  p_act_GBC) / 3

    #concatenating while stacking
    # Stack_X_cv = pd.concat([p_cv_RFC,p_cv_RBF])
    # Stack_X_test = pd.concat([p_test_RFC,p_test_RBF])
    # Stack_X_Actual = (p_act_RFC + p_act_RBF)/2
    # Stack_Y_cv = np.concatenate((Y_cv,Y_cv),axis=0)
    # Stack_Y_test = np.concatenate((Y_test,Y_test),axis=0)

    #checking with only RFC
    Stack_X_cv = p_cv_RFC
    Stack_X_test = p_test_RFC
    Stack_X_Actual = p_act_RFC
    Stack_Y_cv = Y_cv
    Stack_Y_test = Y_test


    # #Call Logistic regression for final stacking
    pred_test_LR , pred_act_LR = LR_Classifier(Stack_X_cv, Stack_X_test, Stack_Y_cv, Stack_Y_test,
                                                Stack_X_Actual,Sample_DS)

    #print(pred_act_LR)

########################################################################################################################
#Get the predictions for actual data set
########################################################################################################################
    #Get the predictions for actual data set
    # preds = clf.predict_proba(Actual_DS.ix[:,'feat_1':])
    # preds = pd.DataFrame(preds, index=Sample_DS.id.values, columns=Sample_DS.columns[1:])
    # preds.to_csv('C:/Python/Others/data/Kaggle/Otto_Product_Classification/Submission_Roshan.csv', index_label='id')

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)