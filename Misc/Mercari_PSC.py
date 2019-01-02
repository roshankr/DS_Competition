import requests
import numpy as np
import scipy as sp
import sys
import gc
import platform
import pandas as pd
from time import time
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
import re
import warnings
from math import sqrt, exp, log
from csv import DictReader
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint as sp_randint
from sklearn import decomposition, pipeline, metrics
from sklearn import preprocessing
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer, TfidfTransformer
from scipy.sparse import hstack
from scipy.sparse import coo_matrix, csr_matrix
from nltk import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import math
import nltk
import os
from sklearn.preprocessing import LabelBinarizer
from gensim.models import Word2Vec, Doc2Vec
import gensim.models

########################################################################################################################
#Caterpillar Tube Pricing
########################################################################################################################
def preprocessor(text, lemmatizer, stemmer, retainNumbers):
    if pd.isnull(text) == True:
        return ""

    if retainNumbers == True:
        #remove any character not in the listed range
        text = re.sub("[^0-9a-zA-Z\.]+", " ", text)
    else:
        text = re.sub("[^a-zA-Z]+", " ", text)

    #remove extra whitespace
    text = re.sub("\s+", " ", text)
    text = text.lower()

    #split text

    # http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer
    if(lemmatizer == True):

        try:
            wordnet_lemmatizer = WordNetLemmatizer()
            newText = ""
            for word in text.split():
                newText = " ".join((newText, wordnet_lemmatizer.lemmatize(word)))
            text = newText.strip(" ")
            #text = wordnet_lemmatizer.lemmatize(text)
        except Exception as e:
            print(e)
            print("--- downloading nltk wordnet corpora")
            import nltk
            nltk.download('wordnet')
            newText = ""

            for word in text.split():
                newText = " ".join((newText, wordnet_lemmatizer.lemmatize(word)))
            text = newText.strip(" ")

    #http://www.nltk.org/howto/stem.html
    if (stemmer == True):
        try:
            stm = SnowballStemmer("english")
            newText = ""
            for word in text.split():
                newText = " ".join((newText, stm.stem(word)))
            text = newText.strip(" ")
        except Exception as e:
            print(e)
            print("--- downloading nltk snowball data")
            import nltk
            nltk.download('snowball_data')
            newText = ""
            for word in text.split():
                newText = " ".join((newText, stm.stem(word)))
            text = newText.strip(" ")

    return text

########################################################################################################################
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the sentence and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

########################################################################################################################
def getAvgFeatureVecs(sentences, model, num_features):
    # Given a set of sentences (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0
    #
    # Preallocate a 2D numpy array, for speed
    sentenceFeatureVecs = np.zeros((len(sentences),num_features),dtype="float32")
    #
    # Loop through the sentences
    for currentSentence in sentences:
       #
       # Print a status message every 1000th sentence
       if counter%10000. == 0.:
           print ("Extract feature for %d of %d" % (counter, len(sentences)))
       #
       # Call the function (defined above) that makes average feature vectors
       sentenceFeatureVecs[counter] = makeFeatureVec(currentSentence, model, num_features)
       #
       # Increment the counter
       counter = counter + 1
    return sentenceFeatureVecs

########################################################################################################################
#RMSLE Scorer
########################################################################################################################
def RMSLE(solution, submission):
    assert len(solution) == len(submission)
    #to_sum = [(math.log(submission[i] + 1) - math.log(solution[i] + 1)) ** 2.0 for i,pred in enumerate(submission)]
    #return (sum(to_sum) * (1.0/len(solution))) ** 0.5

    score = np.sqrt(((np.log(solution+1) - np.log(submission+1)) ** 2.0).mean())
    return score

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
#Cross Validation using different
########################################################################################################################
def Nfold_Cross_Valid(X, y, clfs, wgts):

    print("***************Starting Kfold Cross validation***************")

    scores=[]
    ss = KFold(len(y), n_folds=5,shuffle=True)
    i = 1

    for trainCV, testCV in ss:

        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        y_train = np.log1p(y_train)

        DS_Blend_Pred = np.zeros((X_test.shape[0], len(clfs)))
        for j, clf in enumerate(clfs):
            clf.fit(X_train, y_train)
            model_name = str(clf).split('(')[0]
            y_pred=np.expm1(clf.predict(X_test))
            DS_Blend_Pred[:, j] = y_pred*wgts[j]

            if len(clfs) > 1:
                print("          %d-iteration for...%s... %s " % (i, model_name, str(RMSLE(y_test,y_pred))))

        y_pred = np.sum(DS_Blend_Pred, axis=1)
        scores.append(RMSLE(y_test,y_pred))
        print(" %d-iteration... %s " % (i,scores))
        i = i + 1

    #Average RMSLE
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Text_Munging(New_DS,text_col,num_features,process_type):

    retainNumbers = False
    useStemming = True
    useLemma = False
    useHashing = True
    minDocFreq = 10

    New_DS[text_col] = New_DS[text_col].fillna("-")
    New_DS[text_col] = New_DS[text_col].apply(preprocessor, args=(useLemma, useStemming, retainNumbers))
    train_data = New_DS[text_col].values.ravel()

    if (process_type=="hash") :
        print("using Hashing")
        vectorizer = HashingVectorizer(n_features=num_features,analyzer=u'word',
                               stop_words='english',lowercase=True, alternate_sign=False,norm=None, binary=False,)
        train_data = vectorizer.transform(train_data)
    elif(process_type=="tfidf") :
        print("using TF-IDF")
        # count_vectorizer = CountVectorizer(analyzer=u'word', stop_words='english', min_df=minDocFreq,ngram_range=(1, 1))
        # train_data = count_vectorizer.fit_transform(train_data)
        # tfidf_transformer = TfidfTransformer(use_idf=True)
        # train_data = tfidf_transformer.fit_transform(train_data)
        # actual_data = count_vectorizer.transform(actual_data)
        # actual_data = tfidf_transformer.transform(actual_data)

        vectorizer = TfidfVectorizer(use_idf=True,sublinear_tf=True, analyzer=u'word',stop_words='english',
                                     min_df=minDocFreq,ngram_range=(1, 1),max_features=num_features)
        train_data = vectorizer.fit_transform(train_data)
        vocab = vectorizer.vocabulary_
        print("vocab length :- " + str(len(vocab)))
    elif (process_type=="w2v") :

        num_workers = 8  # Number of threads to run in parallel
        context = 5  # Context window size
        downsampling = 1e-3  # Downsample setting for frequent words

        np.savetxt(os.path.join(file_path, "w2v_train.txt"), New_DS[text_col].values, fmt='%s')

        New_DS[text_col] = New_DS[text_col].apply(nltk.word_tokenize)
        train = New_DS[text_col].tolist()

        #train = df.apply(lambda x: '%s' % list((x.split())))
        model = Word2Vec(min_count=minDocFreq, window=context, size=num_features, sample=downsampling, negative=5,
                     workers=num_workers, seed=1)
        sentences = gensim.models.word2vec.LineSentence(os.path.join(file_path, "w2v_train.txt"))
        model.build_vocab(sentences)

        # for epoch in range(3):
        #     print("Training....Epoch:-", epoch)
        #     #sentences = shuffle(sentences)
        model.train(sentences,total_examples=model.corpus_count, epochs=5)

        print("Creating average feature vecs for training reviews")
        train_data = getAvgFeatureVecs(train, model, num_features)
        print("Word2Vec feature vector size is", np.shape(train_data))


    else:
        print("using CountVectorizer")
        cv = CountVectorizer(min_df=minDocFreq, max_features= num_features)
        train_data = cv.fit_transform(train_data)

    print("output shape is :- " + str(np.shape(train_data)))

    return train_data

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS):

    print("***************Starting Data cleansing***************")
    global col_vals

    y = Train_DS.price.values

    Train_DS  = Train_DS.drop(['train_id','price'], axis = 1)
    Actual_DS = Actual_DS.drop(['test_id'], axis = 1)

    New_DS = pd.concat([Train_DS, Actual_DS])

    len_Train_DS = len(Train_DS)
    len_Actual_DS = len(Actual_DS)

    del Train_DS, Actual_DS
    gc.collect()

    #New_DS['RecId'] = New_DS.index

    # lbl_enc_list = ['name','category_name','brand_name']
    # for column in lbl_enc_list:
    #     print(column + str (New_DS[column].nunique()))
    # sys.exit(0)

    print("Combined Data Shape :- " + str(np.shape(New_DS)))

    Temp_DS = pd.DataFrame()
    cols_required = []
    ####################################################################################################################
    print("Perform Label Encoding")
    lbl_enc_list = ['category_name']
    for column in lbl_enc_list:
        print("Label Encoding for : " + str(column) + " with column : 1" )
        New_DS[column] = New_DS[column].fillna("-")
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(New_DS[column]))
        New_DS[column+"_new"] = lbl.transform(New_DS[column])
        cols_required.append(column+"_new")

    ####################################################################################################################
    print("Perform OneHot Encoding")
    cat_list = ['item_condition_id','shipping','brand_name']
    for column in cat_list:
        dummies = pd.get_dummies(New_DS[column],sparse=True)
        cols_new = [ column+"_"+str(s) for s in list(dummies.columns)]
        print("Onehot Encoding for : " + str(column) + " with column : " + str(len(cols_new)) )
        New_DS[cols_new] = dummies
        #Temp_DS[cols_new] = New_DS[cols_new]
        cols_required.extend(cols_new)

    #New_DS = New_DS.drop(cat_list, axis=1)
    ####################################################################################################################
    Final_Arr = np.array(New_DS[cols_required])
    print("Size before text processing :- " + str(np.shape(Final_Arr)))
    #text_process_list = ['name','category_name','brand_name','item_description']
    #text_process_wgts = [500,500,500,200]

    countvect_process_list = ['name','category_name']
    countvect_process_wgts = [1000,1000]

    for i, column in enumerate(countvect_process_list):
        New_DS[column] = New_DS[column].fillna("Missing")
        text_data = Data_Text_Munging(New_DS,column,num_features=countvect_process_wgts[i],process_type="countvect")
        print("Count Vectorizer for : " + str(column) + " with column : " + str((np.shape(text_data)[1])) )
        Final_Arr = csr_matrix(hstack((Final_Arr, text_data)))

    text_process_list = ['item_description','name']
    text_process_wgts = [1000,1000]

    for i, column in enumerate(text_process_list):
        New_DS[column] = New_DS[column].fillna("Missing")
        text_data = Data_Text_Munging(New_DS,column,num_features=text_process_wgts[i],process_type="w2v")
        print("Text processing for : " + str(column) + " with column : " + str((np.shape(text_data)[1])) )
        Final_Arr = csr_matrix(hstack((Final_Arr, text_data)))

    Train_Arr  = Final_Arr[np.r_[0:len_Train_DS]]
    Actual_Arr = Final_Arr[np.r_[-len_Actual_DS:0]]

    print("Train Shape :- " + str(np.shape(Train_Arr)))
    print("Actual Shape :- " + str(np.shape(Actual_Arr)))

    ####################################################################################################################
    #print("Starting log transforming")
    #Train_Arr = np.log(1+np.asarray(Train_Arr, dtype=np.float32))
    #Actual_Arr = np.log(1+np.asarray(Actual_Arr, dtype=np.float32))

    print("***************Ending Data cleansing***************")

    return Train_Arr, Actual_Arr, y

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def RFR_Regressor(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting Random Forest Regressor***************")
    t0 = time()

    if grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "max_depth": [1, 3, 5,8,10,12,15,20,25,30, None],
                      "max_features": sp_randint(1, 25),
                      "min_samples_split": sp_randint(1, 25),
                      "min_samples_leaf": sp_randint(1, 25),
                      "bootstrap": [True, False]
                     }

        clf = RandomForestRegressor(n_estimators=100)

        #Remove tube_assembly_id after its been used in cross validation
        Train_DS    = np.delete(Train_DS,0, axis = 1)
        Actual_DS   = np.delete(Actual_DS,0, axis = 1)

        # run randomized search
        n_iter_search = 25
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = gini_scorer,cv=10,n_jobs=-1)

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

        #CV:0.2326 , LB: 0.267541
        #CV:0.2309 , LB:0.266273 (with all features)
        #clf = RandomForestRegressor(n_estimators=200)

        #CV:0.2317 , LB: 0.265382 (with all features)
        #clf = RandomForestRegressor(n_estimators=500)

        #for testing purpose
        #New CV:0.2730 , old CV:0.2386 , LB: 0.267567 (with all features)
        clf = RandomForestRegressor(n_estimators=10,n_jobs=-1)

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)

        #Remove tube_assembly_id after its been used in cross validation
        Train_DS    = np.delete(Train_DS,0, axis = 1)
        Actual_DS   = np.delete(Actual_DS,0, axis = 1)

        Train_DS, y = shuffle(Train_DS, y, random_state=42)
        y = np.log1p(y)

        clf.fit(Train_DS, y)

        feature = pd.DataFrame()
        feature['imp'] = clf.feature_importances_
        feature['col'] = col_vals
        feature = feature.sort(['imp'], ascending=False)
        print(feature)

    #Predict actual model
    pred_Actual = np.expm1(clf.predict(Actual_DS))
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_RF_2.csv', index_label='id')

    print("***************Ending Random Forest Regressor***************")
    return pred_Actual

########################################################################################################################
#XGB Regressor
########################################################################################################################
def XGB_Regressor(Train_DS, y, Actual_DS, Sample_DS, grid):

    print("***************Starting xgb Regressor (sklearn)***************")
    t0 = time()

    if grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "n_estimators": [100],
                      "max_depth": sp_randint(1, 30),
                      "min_child_weight": sp_randint(1, 30),
                      "subsample": [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9, 1],
                      "colsample_bytree": [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9, 1],
                      "silent": [True],
                      "gamma": [0.5, 0.6,0.7,0.8,0.9, 1,2]
                     }

        clf = xgb.XGBRegressor()

        # run randomized search
        n_iter_search = 1000
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = RMSLE_scorer,cv=3)

        start = time()
        clf.fit(Train_DS, np.log1p(y))

        print("RandomizedSearchCV took %.2f seconds for %d candidates"
                " parameter settings." % ((time() - start), n_iter_search))
        report(clf.grid_scores_)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        #print(clf.grid_scores_)
        print(clf.best_score_)
        print(clf.best_params_)
        #print(clf.scorer_)
    else:
        #best with (n_estimators=5K
        clf = xgb.XGBRegressor(n_estimators=100,nthread=4)
        #clf = xgb.XGBRegressor(n_estimators=2000, max_depth=3, learning_rate=0.715, nthread=4)
        Ridge1 = Ridge(max_iter=100)

        #Ridge1 = Ridge(max_iter=2000,solver="sag", fit_intercept=False, random_state=666, alpha=1.25)
        #Ridge2 = Ridge(max_iter=2000,solver="lsqr", fit_intercept=False, random_state=666, alpha=1.25)
        #Ridge3 = Ridge(max_iter=2000,solver="sag", fit_intercept=True, random_state=666, alpha=1.25)

        clfs = [clf,Ridge1]
        wgts = [0.9,0.1]

        Nfold_score = Nfold_Cross_Valid(Train_DS, y, clfs, wgts)

        sys.exit(0)

        y = np.log1p(y)
        clf.fit(Train_DS, y)

    #Predict actual model
    pred_Actual = np.expm1(clf.predict(Actual_DS))
    pred_Actual = pred_Actual.clip(min=0)
    print("Actual Model predicted")

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.test_id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_XGB_1.csv', index_label='id')

    print("***************Ending xgb Regressor (sklearn)***************")
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, RMSLE_scorer

    # RMSLE_scorer
    RMSLE_scorer = metrics.make_scorer(RMSLE, greater_is_better = False)

    if(platform.system() == "Windows"):
        file_path = 'C:/Python/Others/data/Mercari_PSC/'
    else:
        file_path = ''

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    #Train_DS      = pd.read_csv(file_path+'train.tsv',sep='\t')
    #Actual_DS     = pd.read_csv(file_path+'test.tsv',sep='\t')

    Train_DS      = pd.read_table(file_path+'train.tsv', engine='c').sample(n=50000, random_state = 21)
    Actual_DS     = pd.read_table(file_path+'test.tsv', engine='c').sample(n=50000, random_state = 21)
    Sample_DS = pd.read_csv(file_path + 'sample_submission.csv', sep=',')

    Train_Arr, Actual_Arr, y =  Data_Munging(Train_DS,Actual_DS)

    #pred_Actual = RFR_Regressor(Train_Arr, y, Actual_Arr, Sample_DS, grid=False)
    pred_Actual = XGB_Regressor(Train_Arr, y, Actual_Arr, Sample_DS, grid=False)
########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)

#last 0.51836