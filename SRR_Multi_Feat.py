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
from collections import Counter
import nltk
from sklearn.metrics import jaccard_similarity_score
from nltk.collocations import *
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer , WordNetLemmatizer

########################################################################################################################
#Search Results Relevance
#Best model as of now is CV Score:', 0.64180189191678194))) and LB : xxxx, (svd=290, C=9,
#after ignoring records > 1.414
########################################################################################################################
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
#Distance metric that takes into account partial agreement when multiple labels are assigned.
#Passonneau 2006, Measuring Agreement on Set-Valued Items (MASI)for Semantic and Pragmatic Annotation.
########################################################################################################################
def masi_distance_new(label1, label2):
    len_intersection = len(set(label1).intersection(label2))
    len_union = len(set(label1).union(label2))
    len_label1 = len(label1)
    len_label2 = len(label2)
    if len_label1 == len_label2 and len_label1 == len_intersection:
        m = 1
    elif len_intersection == min(len_label1, len_label2):
        m = 0.67
    elif len_intersection > 0:
        m = 0.33
    else:
        m = 0

    return 1 - (len_intersection / float(len_union)) * m

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

    #Remove High variance train records
    Train_DS = Train_DS[Train_DS['relevance_variance'] < 1.414].reset_index(drop=True)

    y = Train_DS.median_relevance.values
    idx = Actual_DS.id.values.astype(int)
    Train_DS = Train_DS.drop(['id','median_relevance', 'relevance_variance'], axis = 1)
    Actual_DS = Actual_DS.drop(['id'], axis = 1)

    Train_DS_Q  = Train_DS['query'].str.lower()
    Train_DS_T  = Train_DS['product_title'].str.lower()
    #Train_DS_D  = Train_DS['product_description'].str.lower()
    Actual_DS_Q = Actual_DS['query'].str.lower()
    Actual_DS_T = Actual_DS['product_title'].str.lower()
    #Actual_DS_D = Actual_DS['product_description'].str.lower()

    print("Get length word features for Train and Actual.....")
    #NEW FEATURES - Query length and word count
    Train_DS_Q_len  = Train_DS_Q.str.len().reshape(-1,1)
    Train_DS_Q_wrd  = Train_DS_Q.apply(lambda x: len(re.findall(r'\w+', x))).reshape(-1,1)
    Train_DS_T_len  = Train_DS_T.str.len().reshape(-1,1)
    Train_DS_T_wrd  = Train_DS_T.apply(lambda x: len(re.findall(r'\w+', x))).reshape(-1,1)
    Actual_DS_Q_len = Actual_DS_Q.str.len().reshape(-1,1)
    Actual_DS_Q_wrd = Actual_DS_Q.apply(lambda x: len(re.findall(r'\w+', x))).reshape(-1,1)
    Actual_DS_T_len = Actual_DS_T.str.len().reshape(-1,1)
    Actual_DS_T_wrd = Actual_DS_T.apply(lambda x: len(re.findall(r'\w+', x))).reshape(-1,1)


    print("Get noun features for Train....")
    #New Features - Number of Nouns in Query and title
    Train_DS_Q_NN = []
    Train_DS_T_NN = []
    Train_DS_C_NN = []
    Train_DS_JACC_DIST = []
    Train_DS_EDIT_DIST = []
    Train_DS_INTR_DIST = []
    Train_DS_FREF_DIST = []
    Train_DS_BIGRM = []
    Train_DS_TRIGRM = []

    for i in range(len(Train_DS)):
        if np.mod(i,2000)==0:
            print(" %d-iteration... " % (i))
        tokens_Q = nltk.word_tokenize(Train_DS_Q[i])
        tokens_NN= nltk.pos_tag(tokens_Q)
        propernouns_query = [word for word,pos in tokens_NN if pos.startswith('NN')]
        Train_DS_Q_NN = np.append(Train_DS_Q_NN,len(propernouns_query))

        tokens_T = nltk.word_tokenize(Train_DS_T[i])
        tokens_NN= nltk.pos_tag(tokens_T)
        propernouns_title = [word for word,pos in tokens_NN if pos.startswith('NN')]
        Train_DS_T_NN = np.append(Train_DS_T_NN,len(propernouns_title))

        if len(propernouns_query) > 0:
            Train_DS_C_NN = np.append(Train_DS_C_NN,(float(len(list(set(propernouns_query).intersection(propernouns_title))))/len(propernouns_query)))
        else:
            Train_DS_C_NN = np.append(Train_DS_C_NN,0)

        #jaccard distance
        if len(propernouns_query) > 0:
            Train_DS_JACC_DIST = np.append(Train_DS_JACC_DIST,(len(set(tokens_Q).union(tokens_T)) - len(set(tokens_Q).intersection(tokens_T)))/float(len(set(tokens_Q).union(tokens_T))))
        else:
            Train_DS_JACC_DIST = np.append(Train_DS_JACC_DIST,0)

        Train_DS_EDIT_DIST = np.append(Train_DS_EDIT_DIST,nltk.metrics.distance.edit_distance(tokens_Q, tokens_T))
        Train_DS_INTR_DIST = np.append(Train_DS_INTR_DIST,masi_distance_new(tokens_Q, tokens_T))
        Train_DS_FREF_DIST = np.append(Train_DS_FREF_DIST,nltk.metrics.scores.f_measure(set(tokens_Q), set(tokens_T),alpha=0.5))

        tokens_Q_bigram = list(nltk.bigrams(tokens_Q))
        tokens_T_bigram = list(nltk.bigrams(tokens_T))

        tokens_Q_trigram = list(nltk.trigrams(tokens_Q))
        tokens_T_trigram = list(nltk.trigrams(tokens_T))

        if len(tokens_Q_bigram) > 0:
            #Train_DS_BIGRM = np.append(Train_DS_BIGRM,(len(list(set(tokens_Q_bigram).intersection(tokens_T_bigram))))/len(tokens_Q_bigram))
            Train_DS_BIGRM = np.append(Train_DS_BIGRM,(len(list(set(tokens_Q_bigram).intersection(tokens_T_bigram)))))
        else:
            Train_DS_BIGRM = np.append(Train_DS_BIGRM,0)

        if len(tokens_Q_trigram) > 0:
            #Train_DS_TRIGRM = np.append(Train_DS_TRIGRM,(len(list(set(tokens_Q_trigram).intersection(tokens_T_trigram))))/len(tokens_Q_trigram))
            Train_DS_TRIGRM = np.append(Train_DS_TRIGRM,(len(list(set(tokens_Q_trigram).intersection(tokens_T_trigram)))))
        else:
            Train_DS_TRIGRM = np.append(Train_DS_TRIGRM,0)

    Train_DS_Q_NN = Train_DS_Q_NN.reshape(-1,1)
    Train_DS_T_NN = Train_DS_T_NN.reshape(-1,1)
    Train_DS_C_NN = Train_DS_C_NN.reshape(-1,1)
    Train_DS_JACC_DIST = Train_DS_JACC_DIST.reshape(-1,1)
    Train_DS_EDIT_DIST = Train_DS_EDIT_DIST.reshape(-1,1)
    Train_DS_INTR_DIST = Train_DS_INTR_DIST.reshape(-1,1)
    Train_DS_FREF_DIST = Train_DS_FREF_DIST.reshape(-1,1)
    Train_DS_BIGRM = Train_DS_BIGRM.reshape(-1,1)
    Train_DS_TRIGRM = Train_DS_TRIGRM.reshape(-1,1)

    print("Get noun features for Train....")
    #New Features - Number of Nouns in Query and title
    Actual_DS_Q_NN = []
    Actual_DS_T_NN = []
    Actual_DS_C_NN = []
    Actual_DS_JACC_DIST = []
    Actual_DS_EDIT_DIST = []
    Actual_DS_INTR_DIST = []
    Actual_DS_FREF_DIST = []
    Actual_DS_BIGRM = []
    Actual_DS_TRIGRM = []

    for i in range(len(Actual_DS)):
        if np.mod(i,2000)==0:
            print(" %d-iteration... " % (i))
        tokens_Q = nltk.word_tokenize(Actual_DS_Q[i])
        tokens_NN= nltk.pos_tag(tokens_Q)
        propernouns_query = [word for word,pos in tokens_NN if pos.startswith('NN')]
        Actual_DS_Q_NN = np.append(Actual_DS_Q_NN,len(propernouns_query))

        tokens_T = nltk.word_tokenize(Actual_DS_T[i])
        tokens_NN= nltk.pos_tag(tokens_T)
        propernouns_title = [word for word,pos in tokens_NN if pos.startswith('NN')]
        Actual_DS_T_NN = np.append(Actual_DS_T_NN,len(propernouns_title))

        if len(propernouns_query) > 0:
            Actual_DS_C_NN = np.append(Actual_DS_C_NN,(float(len(list(set(propernouns_query).intersection(propernouns_title))))/len(propernouns_query)))
        else:
            Actual_DS_C_NN = np.append(Actual_DS_C_NN,0)

        #jaccard distance
        if len(propernouns_query) > 0:
            Actual_DS_JACC_DIST = np.append(Actual_DS_JACC_DIST,(len(set(tokens_Q).union(tokens_T)) - len(set(tokens_Q).intersection(tokens_T)))/float(len(set(tokens_Q).union(tokens_T))))
        else:
            Actual_DS_JACC_DIST = np.append(Actual_DS_JACC_DIST,0)

        Actual_DS_EDIT_DIST = np.append(Actual_DS_EDIT_DIST,nltk.metrics.distance.edit_distance(tokens_Q, tokens_T))
        Actual_DS_INTR_DIST = np.append(Actual_DS_INTR_DIST,masi_distance_new(tokens_Q, tokens_T))
        Actual_DS_FREF_DIST = np.append(Actual_DS_FREF_DIST,nltk.metrics.scores.f_measure(set(tokens_Q), set(tokens_T),alpha=0.5))

        tokens_Q_bigram = list(nltk.bigrams(tokens_Q))
        tokens_T_bigram = list(nltk.bigrams(tokens_T))

        tokens_Q_trigram = list(nltk.trigrams(tokens_Q))
        tokens_T_trigram = list(nltk.trigrams(tokens_T))

        if len(tokens_Q_bigram) > 0:
            #Actual_DS_BIGRM = np.append(Actual_DS_BIGRM,(len(list(set(tokens_Q_bigram).intersection(tokens_T_bigram))))/len(tokens_Q_bigram))
            Actual_DS_BIGRM = np.append(Actual_DS_BIGRM,(len(list(set(tokens_Q_bigram).intersection(tokens_T_bigram)))))
        else:
            Actual_DS_BIGRM = np.append(Actual_DS_BIGRM,0)

        if len(tokens_Q_trigram) > 0:
            #Actual_DS_TRIGRM = np.append(Actual_DS_TRIGRM,(len(list(set(tokens_Q_trigram).intersection(tokens_T_trigram))))/len(tokens_Q_trigram))
            Actual_DS_TRIGRM = np.append(Actual_DS_TRIGRM,(len(list(set(tokens_Q_trigram).intersection(tokens_T_trigram)))))
        else:
            Actual_DS_TRIGRM = np.append(Actual_DS_TRIGRM,0)

    Actual_DS_Q_NN = Actual_DS_Q_NN.reshape(-1,1)
    Actual_DS_T_NN = Actual_DS_T_NN.reshape(-1,1)
    Actual_DS_C_NN = Actual_DS_C_NN.reshape(-1,1)
    Actual_DS_JACC_DIST = Actual_DS_JACC_DIST.reshape(-1,1)
    Actual_DS_EDIT_DIST = Actual_DS_EDIT_DIST.reshape(-1,1)
    Actual_DS_INTR_DIST = Actual_DS_INTR_DIST.reshape(-1,1)
    Actual_DS_FREF_DIST = Actual_DS_FREF_DIST.reshape(-1,1)
    Actual_DS_BIGRM = Actual_DS_BIGRM.reshape(-1,1)
    Actual_DS_TRIGRM = Actual_DS_TRIGRM.reshape(-1,1)

    #do some lambda magic on text columns
    Train_DS  = list(Train_DS.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    Actual_DS = list(Actual_DS.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

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
    #Train_DS_D  = tfv.transform(Train_DS_D)
    Actual_DS_Q = tfv.transform(Actual_DS_Q)
    Actual_DS_T = tfv.transform(Actual_DS_T)
    #Actual_DS_D = tfv.transform(Actual_DS_D)

    print("starting SVD conversion...")
    #Setting singular value decomposition
    svd = TruncatedSVD(n_components=290)
    svd.fit(Train_DS)
    Train_DS = svd.transform(Train_DS)
    Actual_DS = svd.transform(Actual_DS)

    Train_DS_Q  = svd.transform(Train_DS_Q)
    Train_DS_T  = svd.transform(Train_DS_T)
    #Train_DS_D  = svd.transform(Train_DS_D)

    Actual_DS_Q = svd.transform(Actual_DS_Q)
    Actual_DS_T = svd.transform(Actual_DS_T)
    #Actual_DS_D = svd.transform(Actual_DS_D)

    dist_models=[
        'braycurtis'
        ,'canberra'
        ,'chebyshev'
        ,'correlation'
        ,'manhattan'
        ,'minkowski'
         ]

    #braycurtis, correlation, minkowski
    print("Start Calculating multiple distance vectors......")
    for dist in dist_models:
        print(dist)
        #jac_dist_T = np.array(pairwise_distances(Train_DS_Q[200:210], Train_DS_T[200:210], metric=dist,n_jobs=-1).diagonal()).reshape(-1,1)
        misc_dist_T = np.array(pairwise_distances(Train_DS_Q, Train_DS_T, metric=dist,n_jobs=-1).diagonal()).reshape(-1,1)
        misc_dist_T = np.nan_to_num(misc_dist_T)
        Train_DS = np.append(Train_DS,misc_dist_T,1)

        misc_dist_A = np.array(pairwise_distances(Actual_DS_Q, Actual_DS_T, metric=dist,n_jobs=-1).diagonal()).reshape(-1,1)
        misc_dist_A = np.nan_to_num(misc_dist_A)
        Actual_DS = np.append(Actual_DS,misc_dist_A,1)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    print("starting euc_dist conversion...")
    euc_dist_T  = np.array(euclidean_distances(Train_DS_Q, Train_DS_T).diagonal()).reshape(-1,1)
    euc_dist_A  = np.array(euclidean_distances(Actual_DS_Q, Actual_DS_T).diagonal()).reshape(-1,1)
    #euc_dist_T1 = np.array(euclidean_distances(Train_DS_Q, Train_DS_D).diagonal()).reshape(-1,1)
    #euc_dist_A1 = np.array(euclidean_distances(Actual_DS_Q, Actual_DS_D).diagonal()).reshape(-1,1)

    print("starting cos_dist conversion...")
    cos_dist_T  = np.array(cosine_distances(Train_DS_Q, Train_DS_T).diagonal()).reshape(-1,1)
    cos_dist_A  = np.array(cosine_distances(Actual_DS_Q, Actual_DS_T).diagonal()).reshape(-1,1)
    #cos_dist_T1 = np.array(cosine_distances(Train_DS_Q, Train_DS_D).diagonal()).reshape(-1,1)
    #cos_dist_A1 = np.array(cosine_distances(Actual_DS_Q, Actual_DS_D).diagonal()).reshape(-1,1)


    print("Add all features to train set...")
    #add any other distance matrix
    #Train_DS_New_features = [euc_dist_T,cos_dist_T,euc_dist_T1,cos_dist_T1,pair_dist_T,pair_dist_T1]
    # Train_DS_New_features = np.concatenate((euc_dist_T,cos_dist_T,euc_dist_T1,cos_dist_T1,
    #                                        Train_DS_Q_len,Train_DS_Q_wrd,Train_DS_T_len,Train_DS_T_wrd,
    #                                        Train_DS_Q_NN,Train_DS_T_NN,Train_DS_C_NN,
    #                                        Train_DS_JACC_DIST,Train_DS_EDIT_DIST,Train_DS_INTR_DIST,
    #                                        Train_DS_FREF_DIST, Train_DS_BIGRM, Train_DS_TRIGRM
    #                                        ),axis=1)

    Train_DS_New_features = np.concatenate((euc_dist_T,cos_dist_T,
                                           Train_DS_Q_len,Train_DS_Q_wrd,Train_DS_T_len,Train_DS_T_wrd,
                                           Train_DS_Q_NN,Train_DS_T_NN,Train_DS_C_NN,
                                           Train_DS_JACC_DIST,Train_DS_EDIT_DIST,Train_DS_INTR_DIST,
                                           Train_DS_FREF_DIST, Train_DS_BIGRM, Train_DS_TRIGRM
                                           ),axis=1)

    Train_DS = np.append(Train_DS,Train_DS_New_features,1)

    print("Add all features to actual set...")
    # Actual_DS_New_features = np.concatenate((euc_dist_A,cos_dist_A,euc_dist_A1,cos_dist_A1,
    #                                        Actual_DS_Q_len,Actual_DS_Q_wrd,Actual_DS_T_len,Actual_DS_T_wrd,
    #                                        Actual_DS_Q_NN,Actual_DS_T_NN,Actual_DS_C_NN,
    #                                        Actual_DS_JACC_DIST,Actual_DS_EDIT_DIST,Actual_DS_INTR_DIST,
    #                                        Actual_DS_FREF_DIST, Actual_DS_BIGRM, Actual_DS_TRIGRM
    #                                        ),axis=1)

    Actual_DS_New_features = np.concatenate((euc_dist_A,cos_dist_A,
                                           Actual_DS_Q_len,Actual_DS_Q_wrd,Actual_DS_T_len,Actual_DS_T_wrd,
                                           Actual_DS_Q_NN,Actual_DS_T_NN,Actual_DS_C_NN,
                                           Actual_DS_JACC_DIST,Actual_DS_EDIT_DIST,Actual_DS_INTR_DIST,
                                           Actual_DS_FREF_DIST, Actual_DS_BIGRM, Actual_DS_TRIGRM
                                           ),axis=1)

    Actual_DS = np.append(Actual_DS,Actual_DS_New_features,1)

    #temp - copy of munged files
    Train_DS1  = pd.DataFrame(Train_DS)
    Train_DS1['y'] = y
    Actual_DS1 = pd.DataFrame(Actual_DS)

    print("starting Standard Scaler conversion...")
    #Setting Standard scaler for data
    stdScaler = StandardScaler()
    stdScaler.fit(Train_DS)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

    Train_DS1.to_csv(file_path+'Train_DS_Input_Model_3_multi_feature_2.csv', index_label='id')
    Actual_DS1.to_csv(file_path+'Actual_DS_Input_Model_3_multi_feature_2.csv', index_label='id')

    print("***************Ending Data cleansing***************")

    return Train_DS, Actual_DS, y

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

    #pred_Actual = clf.predict(Train_DS)
    #Train_DS1['y'] = pred_Actual
    #Train_DS1.to_csv(file_path+'Train_DS1.csv')

    #Predict actual model
    pred_Actual = clf.predict(Actual_DS)
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_SVC_Model_3_multi_feature_2.csv', index_label='id')

    print("***************Ending SVM with rbf Classifier***************")
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path , kappa_scorer

    #nltk.download()

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
    Train_DS =  pd.read_csv(file_path+'train.csv',sep=',',encoding='utf-8').fillna("")
    Actual_DS =  pd.read_csv(file_path+'test.csv',sep=',',encoding='utf-8').fillna("")
    Sample_DS = pd.read_csv(file_path+'sampleSubmission.csv',sep=',')

    global Train_DS1
    Train_DS1 = Train_DS

    if full_run == False:
        print("skipping Data Munging....")
        y = Train_DS.median_relevance.values
        Train_DS =  pd.read_csv(file_path+'Train_DS_Input_Model_3_multi_feature.csv',sep=',')
        Actual_DS =  pd.read_csv(file_path+'Actual_DS_Input_Model_3_multi_feature.csv',sep=',')
    else:
        print("Model C=9, SVD = 290 with manhattan too")
        Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS)

    pred_Actual = SVM_rbf_Classifier(Train_DS, y, Actual_DS, Sample_DS, grid=False)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)
