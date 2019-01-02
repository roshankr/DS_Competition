import numpy as np
np.random.seed(2016)
import os
import glob
import cv2
import datetime
import time
import sys
import pandas as pd
import warnings
from sklearn.utils import shuffle
import platform
from sklearn.cross_validation import KFold,StratifiedShuffleSplit
from keras.models import Sequential,Model
from keras.layers import Input,BatchNormalization,merge
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D,AveragePooling2D
from keras.optimizers import Adam, Adagrad, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf1
from sklearn.metrics import log_loss
from scipy.misc import imread, imresize, imshow
import random
import pickle
from shutil import copy2
import time as tm
from sklearn import preprocessing
from keras.applications.inception_v3 import InceptionV3
from keras.applications import vgg16
#from vgg16bn import Vgg16BN
from models import Vgg16BN, Inception, Resnet50

#For Lasagne
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, ReshapeLayer,LSTMLayer,RecurrentLayer,Conv2DLayer,MaxPool2DLayer
from lasagne.updates import nesterov_momentum,adagrad
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
from nolearn.lasagne import NeuralNet
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from lasagne import layers
from lasagne.nonlinearities import  softmax, rectify
from lasagne.updates import nesterov_momentum,sgd,adagrad,adadelta,rmsprop
from lasagne import nonlinearities as nl
from nolearn.lasagne import BatchIterator
from lasagne.init import GlorotUniform
from lasagne.regularization import *
from glob import iglob

#######################################################################################################################
#train model
########################################################################################################################
def train(parent_model, model_str):
    parent_model.build()
    model_fn = saved_model_path + '{val_loss:.2f}-loss_{epoch}epoch_' + model_str
    ckpt = ModelCheckpoint(filepath=model_fn, monitor='val_loss',
                           save_best_only=True, save_weights_only=True)

    if use_val:
        parent_model.fit_val(split_train_path, valid_path, nb_trn_samples=nb_split_train_samples,
                             nb_val_samples=nb_valid_samples, nb_epoch=nb_epoch, callbacks=[ckpt], aug=nb_aug)

        model_path = max(iglob(saved_model_path + '*.h5'), key=os.path.getctime)
        return model_path

    model_fn = saved_model_path + '{}epoch_'.format(nb_epoch) + model_str
    parent_model.fit_full(full_train_path, nb_trn_samples=nb_full_train_samples, nb_epoch=nb_epoch, aug=nb_aug)
    model.save_weights(model_fn)
    del parent_model.model

    return model_fn
#######################################################################################################################
#train model
########################################################################################################################
def train_all():
    model_paths = {
        "vggbn": [],
        "inception": [],
        'resnet': [],
    }

    for run in range(nb_runs):
        print("Starting Training Run {0} of {1}...\n".format(run+1, nb_runs))
        aug_str = "aug" if nb_aug else "no-aug"

        for arch in archs:
            print("Training {} model...\n".format(arch))
            model = models[arch]
            model_str = "{0}x{1}_{2}_{3}lr_run{4}_{5}.h5".format(model.size[0], model.size[1], aug_str,
                                                                 model.lr, run, arch)
            model_path = train(model, model_str)
            model_paths[arch].append(model_path)

    print("Done.")
    return model_paths

#######################################################################################################################
#train model
########################################################################################################################
def test(model_paths):
    predictions_full = np.zeros((nb_test_samples, nb_classes))

    for run in range(nb_runs):
        print("\nStarting Prediction Run {0} of {1}...\n".format(run+1, nb_runs))
        predictions_aug = np.zeros((nb_test_samples, nb_classes))

        for aug in range(nb_aug):
            print("\n--Predicting on Augmentation {0} of {1}...\n".format(aug+1, nb_aug))
            predictions_mod = np.zeros((nb_test_samples, nb_classes))

            for arch in archs:
                print("----Predicting on {} model...".format(arch))
                parent = models[arch]
                model = parent.build()
                model.load_weights(model_paths[arch][run])
                pred, filenames = parent.test(test_path, nb_test_samples, aug=nb_aug)
                predictions_mod += pred

            predictions_mod /= len(archs)
            predictions_aug += predictions_mod

        predictions_aug /= nb_aug
        predictions_full += predictions_aug

    predictions_full /= nb_runs
    return predictions_full, filenames

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def write_submission(predictions, filenames):
    preds = np.clip(predictions, clip, 1-clip)
    sub_fn = submission_path + '{0}epoch_{1}aug_{2}clip_{3}runs'.format(nb_epoch, nb_aug, clip, nb_runs)

    for arch in archs:
        sub_fn += "_{}".format(arch)

    print(submission_path)
    print(sub_fn)
    with open(sub_fn + '.csv', 'w') as f:
        print("Writing Predictions to CSV...")
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for i, image_name in enumerate(filenames):
            pred = ['%.6f' % p for p in preds[i, :]]
            f.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
        print("Done.")

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging():

    global batch_size,nb_split_train_samples,nb_full_train_samples,nb_valid_samples,nb_test_samples,nb_runs,\
        nb_epoch,nb_aug,dropout,clip,use_val,archs,models, saved_model_path,split_train_path,valid_path,full_train_path,model,nb_classes,test_path,submission_path

    ####################################################################################################################
    #Orig Dataset
    saved_model_path = os.path.join(file_path,'models/')
    split_train_path = os.path.join(file_path,'train','train_split')
    valid_path = os.path.join(file_path,'train','val_split')
    full_train_path = os.path.join(file_path,'train','train_full')
    test_path = os.path.join(file_path,'test_full')
    submission_path = file_path+'subm1/'

    batch_size = 32
    nb_split_train_samples = 3277
    nb_full_train_samples = 3777
    nb_valid_samples = 500
    nb_test_samples = 1000
    ####################################################################################################################

    #Processed DataSet
    # saved_model_path = os.path.join(file_path,'models/')
    # split_train_path = os.path.join(file_path,'train','Processed_train_split')
    # valid_path = os.path.join(file_path,'train','Processed_val_split')
    # full_train_path = os.path.join(file_path,'train','train_full')
    # test_path = os.path.join(file_path,'test_full_Processed')
    # submission_path = file_path+'subm1/'

    # data
    # batch_size = 32
    # nb_split_train_samples = 4659
    # nb_full_train_samples = 4659
    # nb_valid_samples = 500
    # nb_test_samples = 1000

    ####################################################################################################################

    classes = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
    nb_classes = len(classes)

    # orig model
    # nb_runs = 5
    # nb_epoch = 30
    # nb_aug = 5
    # dropout = 0.4
    # clip = 0.01
    # use_val = True

    nb_runs = 2
    nb_epoch = 50
    nb_aug = 10
    dropout = 0.4
    clip = 0.02
    use_val = True
    batch_size = 32

    #archs = ["inception"]
    archs = ["vggbn","inception","resnet"]

    models = {
        "vggbn": Vgg16BN(size=(224, 224), n_classes=nb_classes, lr=0.001,
                           batch_size=batch_size, dropout=dropout),
        "inception": Inception(size=(299, 299), n_classes=nb_classes,
                           lr=0.001, batch_size=batch_size),
        #"resnet": Resnet50(size=(270, 480), n_classes=nb_classes, lr=0.001,
        #            batch_size=batch_size, dropout=dropout)
        "resnet": Resnet50(size=(224, 224), n_classes=nb_classes, lr=0.001,
                    batch_size=batch_size, dropout=dropout)
    }

    model_paths = train_all()
    print(model_paths)

    #vgg path
    #model_paths = {'inception': [], 'vggbn': ['/home/roshan/Desktop/DS/Others/data/Kaggle/Nature_Conservancy_Fisheries_Monitoring/models1.73-loss_0epoch_64x64_aug_0.001lr_run0_vggbn.h5'], 'resnet': []}

    predictions, filenames = test(model_paths)

    write_submission(predictions, filenames)

    sys.exit(0)


########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    global file_path, use_cache, train_folder, test_folder, restore_from_last_checkpoint,\
        img_rows,img_cols,color_type_global,nb_epoch,batch_size,predict_proba,orig_input,data_aug


    #img_rows, img_cols = 64, 64
    color_type_global = 3
    #batch_size = 16
    #nb_epoch = 50
    predict_proba = True


    use_cache = 0
    restore_from_last_checkpoint = 0

    batch_size = 32
    nb_epoch = 1

    #only for googlenet
    img_rows, img_cols = 64, 64
    #img_rows, img_cols = 480, 270
    batch_size = 32
    nb_epoch = 8

    orig_input = 'orig'
    data_aug = False

    train_folder = 'train'
    test_folder = 'test'

    if(platform.system() == "Windows"):

        file_path = 'C:\\Python\\Others\\data\\Nature_Conservancy_Fisheries_Monitoring'

    else:
        #aws
        #file_path = '/DS/Nature_Conservancy_Fisheries_Monitoring/'
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Nature_Conservancy_Fisheries_Monitoring/'

    Data_Munging()

    #Model_building(train_data,train_target,train_target_vect,train_id,test_data,test_id)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)

#Classifier 3 -> only one run with orig = 1.32 (.5 cv)
#Classifier 3 -> only one run with proc = 2.43 (.6 cv)
#Classifier 3 -> only one run with Augm = 1.86 (.8 cv)
#VGG16 pretrained went really bad , overfitted only to few fish categories
#Classifier 3 -> only one run with orig and shape  480, 270 = not good