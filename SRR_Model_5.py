import requests
import numpy as np
import scipy as sp
import sys
import platform
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from nltk.stem.porter import *
from time import time
from scipy.stats import randint as sp_randint
import re
from sklearn.utils import shuffle
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.ensemble import RandomForestClassifier ,ExtraTreesClassifier,AdaBoostClassifier
import warnings
from sklearn.metrics.pairwise import *
from scipy.sparse import csr_matrix, vstack, hstack
from operator import itemgetter
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum,adagrad,sgd
from lasagne.nonlinearities import identity,sigmoid, tanh,rectify
from lasagne.objectives import binary_crossentropy
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
import theano
from theano import tensor as T
from theano.tensor.nnet import softmax

########################################################################################################################
#Search Results Relevance
#Best model as of now is CV Score:', 0.617)) and LB : 0.57 WIT SVD = 400 , AND NO PROD DESC, MANHATTAN
########################################################################################################################
########################################################################################################################
#AdjustVariable for NN
########################################################################################################################
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
########################################################################################################################
#EarlyStopping for NN
########################################################################################################################
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

########################################################################################################################
#Stemming functionality
########################################################################################################################
class stemmerUtility(object):
    """Stemming functionality"""
    @staticmethod
    def stemPorter(review_text):
        porter = PorterStemmer()
        preprocessed_docs = []
        for doc in review_text:
            final_doc = []
            for word in doc:
                final_doc.append(porter.stem(word))
                #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
            preprocessed_docs.append(final_doc)
        return preprocessed_docs

########################################################################################################################
#The following 3 functions have been taken from Ben Hamner's github repository [https://github.com/benhamner/Metrics]
#Returns the confusion matrix between rater's ratings
########################################################################################################################
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

########################################################################################################################
#Returns the counts of each type of rating that a rater made
########################################################################################################################
def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

########################################################################################################################
#Calculates the quadratic weighted kappa
########################################################################################################################
def quadratic_weighted_kappa(y, y_pred):

    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

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
        y_pred=clf.predict(X_test)
        scores.append(quadratic_weighted_kappa(y_test, y_pred))
        print(" %d-iteration... %s " % (i,scores))
        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS):

    print("***************Starting Data cleansing***************")

    stemmer = PorterStemmer()

    # array declarations
    sw=[]
    s_data = []
    s_labels = []
    t_data = []
    t_labels = []

    #stopwords tweak - more overhead
    stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
    #stop_words = ['http','www','img','border','color','style','padding','table','font','thi','inch','ha','width','height',
    #              '0','1','2','3','4','5','6','7','8','9']
    stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

    for stw in stop_words:
        sw.append("q"+stw)
        sw.append("z"+stw)
    stop_words = text.ENGLISH_STOP_WORDS.union(sw)

    # for i in range(len(stop_words)):
    #     stop_words[i]=stemmer.stem(stop_words[i])

    y = Train_DS.median_relevance.values
    idx = Actual_DS.id.values.astype(int)
    Train_DS = Train_DS.drop(['id','median_relevance', 'relevance_variance'], axis = 1)
    Actual_DS = Actual_DS.drop(['id'], axis = 1)

    Train_DS_Q  = Train_DS['query']
    Train_DS_T  = Train_DS['product_title']
    Train_DS_D  = Train_DS['product_description']
    Actual_DS_Q = Actual_DS['query']
    Actual_DS_T = Actual_DS['product_title']
    Actual_DS_D = Actual_DS['product_description']

    #do some lambda magic on text columns
    Train_DS  = list(Train_DS.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    Actual_DS = list(Actual_DS.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

    #Train_DS = list(Train_DS.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1))
    #Actual_DS = list(Actual_DS.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1))

    print("starting TFID conversion...")
    tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

    # Fit TFIDF
    #tfv.fit(np.concatenate((Train_DS, Actual_DS), axis=0))
    tfv.fit(Train_DS)
    Train_DS =  tfv.transform(Train_DS)
    Actual_DS = tfv.transform(Actual_DS)

    Train_DS_Q  = tfv.transform(Train_DS_Q)
    Train_DS_T  = tfv.transform(Train_DS_T)
    Train_DS_D  = tfv.transform(Train_DS_D)
    Actual_DS_Q = tfv.transform(Actual_DS_Q)
    Actual_DS_T = tfv.transform(Actual_DS_T)
    Actual_DS_D = tfv.transform(Actual_DS_D)

    print("starting euc_dist conversion...")
    euc_dist_T  = np.array(euclidean_distances(Train_DS_Q, Train_DS_T).diagonal()).reshape(-1,1)
    euc_dist_A  = np.array(euclidean_distances(Actual_DS_Q, Actual_DS_T).diagonal()).reshape(-1,1)
    euc_dist_T1 = np.array(euclidean_distances(Train_DS_Q, Train_DS_D).diagonal()).reshape(-1,1)
    euc_dist_A1 = np.array(euclidean_distances(Actual_DS_Q, Actual_DS_D).diagonal()).reshape(-1,1)

    print("starting cos_dist conversion...")
    cos_dist_T  = np.array(cosine_distances(Train_DS_Q, Train_DS_T).diagonal()).reshape(-1,1)
    cos_dist_A  = np.array(cosine_distances(Actual_DS_Q, Actual_DS_T).diagonal()).reshape(-1,1)
    cos_dist_T1 = np.array(cosine_distances(Train_DS_Q, Train_DS_D).diagonal()).reshape(-1,1)
    cos_dist_A1 = np.array(cosine_distances(Actual_DS_Q, Actual_DS_D).diagonal()).reshape(-1,1)

    print("starting man_dist conversion 1...")
    pair_dist_T  = np.array(pairwise_distances(Train_DS_Q, Train_DS_T,   metric='manhattan',n_jobs=-1).diagonal()).reshape(-1,1)
    print("starting man_dist conversion 2...")
    pair_dist_A  = np.array(pairwise_distances(Actual_DS_Q, Actual_DS_T, metric='manhattan',n_jobs=-1).diagonal()).reshape(-1,1)
    print("starting man_dist conversion 3...")
    pair_dist_T1 = np.array(pairwise_distances(Train_DS_Q, Train_DS_D,   metric='manhattan',n_jobs=-1).diagonal()).reshape(-1,1)
    print("starting man_dist conversion 4...")
    pair_dist_A1 = np.array(pairwise_distances(Actual_DS_Q, Actual_DS_D, metric='manhattan',n_jobs=-1).diagonal()).reshape(-1,1)

    print("starting SVD conversion...")
    #Setting singular value decomposition
    svd = TruncatedSVD(n_components=400)
    svd.fit(Train_DS)
    Train_DS = svd.transform(Train_DS)
    Actual_DS = svd.transform(Actual_DS)

    Train_DS = np.append(Train_DS, euc_dist_T,1)
    Train_DS = np.append(Train_DS, cos_dist_T,1)
    Train_DS = np.append(Train_DS, pair_dist_T,1)
    Train_DS = np.append(Train_DS, euc_dist_T1,1)
    Train_DS = np.append(Train_DS, cos_dist_T1,1)
    Train_DS = np.append(Train_DS, pair_dist_T1,1)

    Actual_DS = np.append(Actual_DS, euc_dist_A,1)
    Actual_DS = np.append(Actual_DS, cos_dist_A,1)
    Actual_DS = np.append(Actual_DS, pair_dist_A,1)
    Actual_DS = np.append(Actual_DS, euc_dist_A1,1)
    Actual_DS = np.append(Actual_DS, cos_dist_A1,1)
    Actual_DS = np.append(Actual_DS, pair_dist_A1,1)

    print("starting Standard Scaler conversion...")
    #Setting Standard scaler for data
    stdScaler = StandardScaler()
    stdScaler.fit(Train_DS)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

    Train_DS1  = pd.DataFrame(Train_DS)
    Actual_DS1 = pd.DataFrame(Actual_DS)

    Train_DS1.to_csv(file_path+'Train_DS_Input_Model_5.csv')
    Actual_DS1.to_csv(file_path+'Actual_DS_Input_Model_5.csv')

    print("***************Ending Data cleansing***************")

    return Train_DS, Actual_DS, y

########################################################################################################################
#Random Forest Classifier
########################################################################################################################
def RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting RFC Classifier***************")

    t0 = time()

    if grid:

        #use SVD (similar to PCA)
        svd = TruncatedSVD( algorithm='randomized', n_iter=5, random_state=None, tol=0.0)

        # Initialize the standard scaler
        scl = StandardScaler(copy=True, with_mean=True, with_std=True)

       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid/Random Search")

        RFC_model = RandomForestClassifier(n_estimators=500,n_jobs=-1)

        # Create the pipeline
        clf = pipeline.Pipeline([('svd', svd),
    	    					 ('scl', scl),
                        	     ('RFC', RFC_model)])

        # specify parameters and distributions to sample from
        param_dist = {
                      "svd__n_components" : [200,300,400,500,600,700],
                      "max_depth": [1, 2, 3, 4, 5, None],
                      "max_features": sp_randint(1, 40),
                      "min_samples_split": sp_randint(1, 20),
                      "min_samples_leaf": sp_randint(1, 20),
                      "bootstrap": [True, False]
                     }


        # clf = GridSearchCV(estimator = clf, param_grid=param_dist, scoring=kappa_scorer,
        #                              verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

        # run randomized search
        n_iter_search = 1000
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = kappa_scorer,cv=10)

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

        #Setting singular value decomposition
        # svd = TruncatedSVD(n_components=500,algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
        # svd.fit(Train_DS)
        # Train_DS = svd.transform(Train_DS)
        # Actual_DS = svd.transform(Actual_DS)
        #
        # #Setting Standard scaler for data
        # stdScaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        # stdScaler.fit(Train_DS,y)
        # Train_DS = stdScaler.transform(Train_DS)
        # Actual_DS = stdScaler.transform(Actual_DS)

        clf = RandomForestClassifier(n_jobs=-1, n_estimators=500, min_samples_split=1)
        clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict(Actual_DS)
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_RFC.csv', index_label='id')

    print("***************Ending RFC Classifier***************")
    return pred_Actual

########################################################################################################################
#SVM with rbf  Classifier
########################################################################################################################
def SVM_rbf_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting SVM with rbf Classifier***************")

    t0 = time()

    if grid:

        #use SVD (similar to PCA)
        #svd = TruncatedSVD( algorithm='randomized', n_iter=5, random_state=None, tol=0.0)

        # Initialize the standard scaler
        #scl = StandardScaler(copy=True, with_mean=True, with_std=True)

       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid/Random Search")

        clf = SVC(kernel='rbf', max_iter=-1)

        # Create the pipeline
        # clf = pipeline.Pipeline([('scl', scl),
        #                 	     ('svm', svm_model)])

        # specify parameters and distributions to sample from
        param_dist = {
                      #'svd__n_components' : [200,300,400,500,600,700],
                      'C'            : [5,7,8,9,10,11],
                      'gamma'        : [0.01, 0.05, 0.1, 1,0.00]
                     }

        clf = GridSearchCV(estimator = clf, param_grid=param_dist, scoring=kappa_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)

        # run randomized search
        # n_iter_search = 250
        # clf = RandomizedSearchCV(clf, param_distributions=param_dist,
        #                                    n_iter=n_iter_search, scoring = kappa_scorer,cv=10)

        start = time()
        clf.fit(Train_DS, y)

        # print("RandomizedSearchCV took %.2f seconds for %d candidates"
        #          " parameter settings." % ((time() - start), n_iter_search))

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        print(clf.grid_scores_)
        print(clf.best_score_)
        print(clf.best_params_)
        print(clf.scorer_)

        report(clf.grid_scores_)

        # Get best model
        clf = clf.best_estimator_

        # Fit model with best parameters optimized for quadratic_weighted_kappa
        clf.fit(Train_DS, y)

    else:

        #Setting singular value decomposition
        # svd = TruncatedSVD(n_components=400,algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
        # svd.fit(Train_DS)
        # Train_DS = svd.transform(Train_DS)
        # Actual_DS = svd.transform(Actual_DS)
        #
        # #Setting Standard scaler for data
        # stdScaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        # stdScaler.fit(Train_DS,y)
        # Train_DS = stdScaler.transform(Train_DS)
        # Actual_DS = stdScaler.transform(Actual_DS)

        #cv = 0.567
        clf = SVC(C=9, max_iter=-1, random_state=None)

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = clf.predict(Actual_DS)
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_SVC_Model_3.csv', index_label='id')

    print("***************Ending SVM with rbf Classifier***************")
    return pred_Actual

########################################################################################################################
#Neural Network Classifier 1
########################################################################################################################
def NN1_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting NN1 Classifier***************")
    t0 = time()


    if grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

    else:
        #y = y.reshape((-1, 1))
        y   = y.astype('int32')
        Actual_DS  = np.array(Actual_DS.astype('float32'))
        Train_DS   = np.array(Train_DS.astype('float32'))

        learning_rate = theano.shared(np.float32(0.1))
        #Define Model parms - 2 hidden layers
        clf = NeuralNet(
        	layers=[
                    ('input', InputLayer),
                    ('dropout0', DropoutLayer),
                    ('hidden1', DenseLayer),
                    ('dropout1', DropoutLayer),
                    ('hidden2', DenseLayer),
                    ('dropout2', DropoutLayer),
                    ('output', DenseLayer),
       		       ],

   	    # layer parameters:
        input_shape=(None, Train_DS.shape[1]),
        dropout0_p=0.25,
        hidden1_num_units=400,
        dropout1_p = 0.4,
        hidden2_num_units=400,
        dropout2_p = 0.4,

        output_nonlinearity=softmax,  # output layer uses identity function
        output_num_units=5,

        #optimization method
        #update=sgd,
        #update=nesterov_momentum,
        update=adagrad,
        update_learning_rate=0.01,
        use_label_encoder=False,
        batch_iterator_train=BatchIterator(batch_size=100),
        #update_momentum=0.1,
        # on_epoch_finished=[
        # AdjustVariable('update_learning_rate', start=0.1, stop=0.0001),
        # EarlyStopping(patience=10),
        # ],
        eval_size = 0.1,
        regression=False,
        max_epochs=60,
        verbose=1
        )

        Train_DS, y = shuffle(Train_DS, y, random_state=42)
        clf.fit(Train_DS, y)

        _, X_valid, _, y_valid = clf.train_test_split(Train_DS, y, clf.eval_size)

        y_pred=clf.predict(X_valid)
        score=quadratic_weighted_kappa(y_valid, y_pred)

        print("Best score: %0.3f" % score)

    #Predict actual model
    pred_Actual = clf.predict(Actual_DS)
    print("Actual NN1 Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_NN_Model_1.csv', index_label='id')

    print("***************Ending NN1 Classifier***************")
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path , kappa_scorer

    if(platform.system() == "Windows"):
        file_path = 'C:/Python/Others/data/Kaggle/Search_Results_Relevance/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Search_Results_Relevance/'

    # Kappa Scorer
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)

    full_run = True
########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS =  pd.read_csv(file_path+'train.csv',sep=',').fillna("")
    Actual_DS =  pd.read_csv(file_path+'test.csv',sep=',').fillna("")
    Sample_DS = pd.read_csv(file_path+'sampleSubmission.csv',sep=',')

    if full_run == False:
        print("skipping Data Munging....")
        y = Train_DS.median_relevance.values
        Train_DS =  pd.read_csv(file_path+'Train_DS_Input_Model_5.csv',sep=',')
        Actual_DS =  pd.read_csv(file_path+'Actual_DS_Input_Model_5.csv',sep=',')
    else:
        print("Model C=9, SVD = 400")
        Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS)

    #pred_Actual = SVM_rbf_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    #pred_Actual = RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
    pred_Actual = NN1_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)
########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)
