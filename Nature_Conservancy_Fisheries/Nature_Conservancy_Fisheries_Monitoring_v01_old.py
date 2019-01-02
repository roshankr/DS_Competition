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
from sklearn.metrics import log_loss
from scipy.misc import imread, imresize, imshow
import random
import pickle
from shutil import copy2
import time as tm
from sklearn import preprocessing

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
from keras.preprocessing.image import ImageDataGenerator

#######################################################################################################################
#Deep Learning Example with Keras and Lasagne
########################################################################################################################

def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation='relu',
                      border_mode=border_mode,
                      name=conv_name,
                      dim_ordering='th')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x

########################################################################################################################
#Class EarlyStopping for Lasagne NN
########################################################################################################################
class EarlyStopping_Lasagne(object):
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
#Class AdjustVariable for Lasagne NN
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
#Read image using open cv and convert to array
########################################################################################################################
def get_im_cv2_mod(path, img_rows, img_cols, color_type):

    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)

    #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()

    # Reduce size
    #rotate = random.uniform(-10, 10)
    #M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
    #img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    resized = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)

    return resized

########################################################################################################################
#Read and Load test data from subfolders
########################################################################################################################
def load_test(img_rows, img_cols, color_type):
    X_test = []
    X_test_id = []
    start_time = time.time()

    #test_sub_folder = ['test_stg1']
    test_sub_folder = ['test_stg1_Processed']

    print('Read test images')
    for tf in test_sub_folder:
        path = os.path.join(file_path, test_folder, tf, '*_0.jpg')
        files = glob.glob(path)
        print('Load folder %s , Total files :- %d' %(format(tf),len(files)))

        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
            X_test.append(img)
            X_test_id.append(flbase)

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id

########################################################################################################################
#Read and Load train data from subfolders
########################################################################################################################
def load_train(img_rows, img_cols, color_type):
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    #train_sub_folder = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
    #train_sub_folder = ['LAG','BET','DOL']
    #train_sub_folder = ['NoF_Processed','ALB_Processed','BET_Processed','DOL_Processed','LAG_Processed'
    #                    ,'OTHER_Processed','SHARK_Processed','YFT_Processed']

    train_sub_folder = ['LAG_Processed','BET_Processed','DOL_Processed']

    print('Read train images')
    for tf in train_sub_folder:
        path = os.path.join(file_path, train_folder, tf, '*.jpg')

        if tf == 'NoF_Processed':
            path = os.path.join(file_path, train_folder, tf, '*.png')

        files = glob.glob(path)
        print('Load folder %s , Total files :- %d' %(format(tf),len(files)))

        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(tf)

            path = os.path.join(file_path, train_folder, 'preview')
            i = 0
            img_temp = img.reshape((1,) + img.shape)
            for batch in datagen.flow(img, batch_size=1, save_to_dir=path, save_prefix='test',
                                      save_format='jpg'):
                i += 1
                if i > 20:
                    break

    path = os.path.join(file_path, train_folder,'preview')

    for img_cnt in range(0, len(y_train)-1):
        i = 0
        for batch in datagen.flow(X_train[img_cnt], batch_size=1,save_to_dir=path, save_prefix='test', save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

    sys.exit(0)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id

########################################################################################################################
#Read the Test images and convert to arrays
########################################################################################################################
def Get_test_data(img_rows, img_cols, color_type):

    test_data, test_id = load_test(img_rows, img_cols, color_type)

    test_data = np.array(test_data, dtype=np.uint8)

    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))

    # test_data = test_data.swapaxes(3, 1)
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')

    return test_data, test_id

########################################################################################################################
#Read the Train images and convert to arrays
########################################################################################################################
def Get_train_data(img_rows, img_cols, color_type):

    global lbl_y

    train_data, train_target, train_id = load_train(img_rows, img_cols, color_type)

    train_data = np.array(train_data, dtype=np.uint8)

    lbl_y = preprocessing.LabelEncoder()
    lbl_y.fit(list(train_target))
    train_target = lbl_y.transform(train_target)
    train_target_vect = train_target

    train_target = np_utils.to_categorical(train_target)
    train_target = np.array(train_target, dtype=np.uint8)

    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))

    train_data = train_data.astype('float32')
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')

    return train_data, train_target, train_target_vect, train_id


########################################################################################################################
#Inception Network
#https://4.bp.blogspot.com/-TMOLlkJBxms/Vt3HQXpE2cI/AAAAAAAAA8E/7X7XRFOY6Xo/s1600/image03.png
########################################################################################################################
def CNN_Inceptionv03_Classifier(img_rows, img_cols, color_type,num_category):

    predict_proba  = False
    input_shape = (color_type, img_rows, img_cols)
    img_input = Input(shape=input_shape)

    if K.image_dim_ordering() == 'th':
        channel_axis = 1
    else:
        channel_axis = 3

    #channel_axis = 1
    x = conv2d_bn(img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid',)
    x = conv2d_bn(x, 32, 3, 3, border_mode='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2),dim_ordering='th')(x)
    x = conv2d_bn(x, 80, 1, 1, border_mode='valid')
    x = conv2d_bn(x, 192, 3, 3, border_mode='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2),dim_ordering='th')(x)

    # parallel 1 , 2 and 3
    for i in range(3):
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same',dim_ordering='th')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

        x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(i))


    # parallel 4 (1-3-1 and merge it)
    branch3x3 = conv2d_bn(x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,
                             subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2),dim_ordering='th')(x)
    x = merge([branch3x3, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed3')

    # # parallel 5 (1-3-5-3 and merge it)
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same',dim_ordering='th')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed4')

    # # parallel 6,7 (1-3-5-3 and merge it)
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same',dim_ordering='th')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(5 + i))
    #
    # # parallel 8 (1-3-5-3 and merge it)
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same',dim_ordering='th')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed7')
    #
    # parallel 9 (1-4-1 and merge it)
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          subsample=(2, 2), border_mode='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
                            subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2),dim_ordering='th')(x)
    #branch_pool = AveragePooling2D((3, 3), strides=(2, 2),dim_ordering='th')(x)
    x = merge([branch3x3, branch7x7x3, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed8')

    # parallel 10,11 (1-4-5-2 and merge it)
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = merge([branch3x3_1, branch3x3_2],
                          mode='concat', concat_axis=channel_axis,
                          name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
                             mode='concat', concat_axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same',dim_ordering='th')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(9 + i))

    # Classification block
    #x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1000, activation='softmax', name='predictions1')(x)
    x = Dense(num_category, activation='softmax', name='predictions')(x)

    model = Model(input =img_input, output=x, name='inception_v3')
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

########################################################################################################################
#Set up parms for Keras CNN model 2
########################################################################################################################
def CNN_VGG16_Classifier(img_rows, img_cols, color_type,num_category):

    predict_proba  = False
    input_shape = (color_type, img_rows, img_cols)
    img_input = Input(shape=input_shape)

    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1',dim_ordering='th')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2',dim_ordering='th')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',dim_ordering='th')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1',dim_ordering='th')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2',dim_ordering='th')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',dim_ordering='th')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1',dim_ordering='th')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2',dim_ordering='th')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3',dim_ordering='th')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',dim_ordering='th')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1',dim_ordering='th')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2',dim_ordering='th')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3',dim_ordering='th')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',dim_ordering='th')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1',dim_ordering='th')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2',dim_ordering='th')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3',dim_ordering='th')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool',dim_ordering='th')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_category, activation='softmax', name='predictions')(x)

    model = Model(input =img_input, output=x, name='vgg16')
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

########################################################################################################################
#Set up parms for Keras CNN model 2
########################################################################################################################
def CNN_Classifier2(img_rows, img_cols, color_type,num_category):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type, img_rows, img_cols), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_category, activation='softmax'))

    optimizer = Adam(lr=1e-3)
    #optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')


    return model

########################################################################################################################
#Set up parms for Keras CNN model 1
########################################################################################################################
def CNN_Classifier1(img_rows, img_cols, color_type,num_category):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',
                            input_shape=(color_type, img_rows, img_cols),activation='relu',dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal',activation='relu',dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',activation='relu',dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(8, 8),dim_ordering='th'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(num_category))
    model.add(Activation('softmax'))

    #optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adagrad(lr=1e-3, epsilon=1e-08)
    #optimizer = Adam(lr=1e-3)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return model

########################################################################################################################
#Set up parms for Lasagne CNN model
########################################################################################################################
def CNN_Classifier_Lasagne(img_rows, img_cols, color_type,num_category):

    #Define Model parms - 2 hidden layers
    model = NeuralNet(
        	layers=[
                    ('input', InputLayer),
                    ('conv1', Conv2DLayer),
                    ('pool1', MaxPool2DLayer),
                    ('drop1', DropoutLayer),
                    ('conv2', Conv2DLayer),
                    ('pool2', MaxPool2DLayer),
                    ('drop2', DropoutLayer),
                    ('conv3', Conv2DLayer),
                    ('pool3', MaxPool2DLayer),
                    ('drop3', DropoutLayer),
                    #('hidden1', DenseLayer),
                    #('hidden2', DenseLayer),
                    ('output', DenseLayer)
       		       ],

    input_shape=(None, 1,img_rows, img_cols),
    conv1_num_filters=32,conv1_filter_size=(3, 3),conv1_nonlinearity=rectify,conv1_W=GlorotUniform(),pool1_pool_size=(2, 2),
    conv2_num_filters=64,conv2_filter_size=(3, 3),conv2_nonlinearity=rectify,conv2_W=GlorotUniform(),pool2_pool_size=(2, 2),
    conv3_num_filters=128,conv3_filter_size=(3, 3),conv3_nonlinearity=rectify,conv3_W=GlorotUniform(),pool3_pool_size=(2, 2),
    drop1_p=0.50,
    drop2_p=0.50,
    drop3_p=0.50,
    #hidden1_num_units=500,hidden1_nonlinearity=rectify,
    #hidden2_num_units=500,hidden2_nonlinearity=rectify,
    output_num_units=num_category, output_nonlinearity=softmax,

    #update=adagrad,
    update=nesterov_momentum,
    update_learning_rate=theano.shared(np.float32(0.001)),
    update_momentum=theano.shared(np.float32(0.9)),

    objective_loss_function=categorical_crossentropy,

    max_epochs=nb_epoch,
    eval_size=0.2,
    regression=False,
    verbose=1,

    ## batch_iterator_train default is 128
    batch_iterator_train=BatchIterator(batch_size=batch_size),
    batch_iterator_test=BatchIterator(batch_size=batch_size),

    on_epoch_finished=[
            #AdjustVariable('update_learning_rate', start=0.001, stop=0.0001),
            # AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping_Lasagne(patience=2)
        ]

    )

    return model

########################################################################################################################
#Merge predicted outputs from multiple folds (simple avg ensembling)
########################################################################################################################
def Merge_CV_folds_mean(data, nfolds):

    print("Merge predicted outputs....")
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

########################################################################################################################
#Create final output file (after classification) -- kaggle style
########################################################################################################################
def Create_final_output_file(predictions, test_id):

    print("Create final predicted dataset....")
    temp_pred = pd.DataFrame(predictions)
    pred_DF = pd.DataFrame(predictions,columns=lbl_y.inverse_transform(temp_pred.columns))
    pred_DF.insert(0, 'image', test_id)

    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')

    suffix = 'NCFM_'+ str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join(file_path,'subm', 'submission_' + suffix + '.csv')
    pred_DF.to_csv(sub_file, index=False)

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging():

    global datagen
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    print("Starting Train feature creation....... at Time: %s" % (tm.strftime("%H:%M:%S")))
    train_data, train_target, train_target_vect, train_id = Get_train_data(img_rows, img_cols,color_type_global)
    print("Ending Train feature creation....... at Time: %s" % (tm.strftime("%H:%M:%S")))

    print("Starting Test feature creation....... at Time: %s" % (tm.strftime("%H:%M:%S")))
    test_data, test_id = Get_test_data(img_rows, img_cols, color_type_global)
    print("Ending Test feature creation....... at Time: %s" % (tm.strftime("%H:%M:%S")))

    return train_data,train_target,train_target_vect,train_id,test_data,test_id

########################################################################################################################
#Cross Validation and model fitting for Keras model
########################################################################################################################
def Nfold_Cross_Valid(X, y,Xtest, nfolds,num_category):

    print("Starting Model Cross Validation....... at Time: %s" %(tm.strftime("%H:%M:%S")))
    random_state = 51

    yfull_train = dict()
    yfull_test = []
    num_fold = 0
    sum_score = 0

    X =np.array(X)
    scores=[]

    #ss = StratifiedShuffleSplit(y, n_iter=nfolds, test_size=(1.0/nfolds))
    ss = KFold(len(y), n_folds=nfolds,shuffle=True,random_state=random_state)

    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        Y_train, Y_test= y[trainCV], y[testCV]

        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_test), len(Y_test))

        clf = CNN_Classifier2(img_rows, img_cols, color_type_global,num_category)
        #model = CNN_Inceptionv03_Classifier(img_rows, img_cols, color_type_global,num_category)

        kfold_weights_path = os.path.join(file_path, 'cache', 'weights_kfold_' + str(num_fold) + '.h5')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
            ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
        ]

        clf.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  shuffle=True, verbose=1, validation_data=(X_test, Y_test),
                  callbacks=callbacks)

        if predict_proba:
            Y_pred=clf.predict_proba(X_test,batch_size=batch_size, verbose=1)
        else:
            Y_pred = clf.predict(X_test, batch_size=batch_size, verbose=1)

        scores.append(log_loss(Y_test, Y_pred))

        # Store test predictions

        if predict_proba:
            test_prediction = clf.predict_proba(Xtest, batch_size=batch_size, verbose=1)
        else:
            test_prediction = clf.predict(Xtest, batch_size=batch_size, verbose=1)

        yfull_test.append(test_prediction)

        print(" %d-iteration... %s " % (i,scores))

        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))


    print("***************Ending Kfold Cross validation***************")

    return yfull_test

########################################################################################################################
#Cross Validation and model fitting for lasagne
########################################################################################################################
def Nfold_Cross_Valid_Lasagne(X, y,Xtest, nfolds):

    print("Starting Model Cross Validation....... at Time: %s" %(tm.strftime("%H:%M:%S")))
    random_state = 51

    yfull_train = dict()
    yfull_test = []
    num_fold = 0
    sum_score = 0

    X =np.array(X)
    scores=[]

    y = y.astype(np.int32)
    X = X.astype('float32')

    ss = StratifiedShuffleSplit(y, n_iter=nfolds, test_size=(1.0/nfolds))
    #ss = KFold(len(y), n_folds=nfolds,shuffle=True,random_state=random_state)

    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        Y_train, Y_test= y[trainCV], y[testCV]

        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_test), len(Y_test))

        clf = CNN_Classifier_Lasagne(img_rows, img_cols, color_type_global,len(np.unique(y)))
        clf.fit(X_train, Y_train)

        Y_pred=clf.predict_proba(X_test)

        scores.append(log_loss(Y_test, Y_pred))

        # Store test predictions
        test_prediction = clf.predict_proba(Xtest)
        yfull_test.append(test_prediction)

        print("---------------> %d-iteration... %s " % (i,scores))

        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))


    print("***************Ending Kfold Cross validation***************")

    return yfull_test


########################################################################################################################
#Model building and cross validation
########################################################################################################################
def Model_building(train_data,train_target,train_target_vect,train_id,test_data,test_id):

    nfolds=5
    model_type = 'keras'

    num_category = len(pd.DataFrame(train_target).columns)
    print("Number of categories to predict: "+ str(num_category))

    train_data, train_target = shuffle(train_data, train_target)

    # if K.image_dim_ordering() == 'th':
    #     print("using Theano model")
    #     train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    #     test_data  =  test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     print("using Tensorflow model")
    #     train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
    #     test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)

    if model_type == 'keras':

        #model = CNN_Inceptionv03_Classifier(img_rows, img_cols, color_type_global, num_category)
        #model = CNN_VGG16_Classifier(img_rows, img_cols, color_type_global,num_category)
        #model = CNN_Classifier2(img_rows, img_cols, color_type_global,num_category)
        yfull_test = Nfold_Cross_Valid(train_data, train_target,test_data,nfolds,num_category)
    else:
        yfull_test = Nfold_Cross_Valid_Lasagne(train_data, train_target_vect,test_data, nfolds)

    test_res = Merge_CV_folds_mean(yfull_test, nfolds)

    Create_final_output_file(test_res, test_id)

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, use_cache, train_folder, test_folder, restore_from_last_checkpoint,\
        img_rows,img_cols,color_type_global,nb_epoch,batch_size,predict_proba

    img_rows, img_cols = 224, 224
    color_type_global = 3
    #batch_size = 16
    batch_size = 16
    #nb_epoch = 50
    nb_epoch = 25
    predict_proba = True

    use_cache = 0
    restore_from_last_checkpoint = 0

    train_folder = 'train'
    test_folder = 'test'

    if(platform.system() == "Windows"):

        file_path = 'C:\\Python\\Others\\data\\Nature_Conservancy_Fisheries_Monitoring'

    else:
        file_path = '/mnt/hgfs/Python/Others/data/Nature_Conservancy_Fisheries_Monitoring/'

    train_data, train_target, train_target_vect, train_id, test_data, test_id = Data_Munging()

    Model_building(train_data,train_target,train_target_vect,train_id,test_data,test_id)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)