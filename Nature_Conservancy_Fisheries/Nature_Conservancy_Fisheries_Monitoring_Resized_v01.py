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
import platform
from sklearn.cross_validation import KFold,StratifiedShuffleSplit
from sklearn.metrics import log_loss
from scipy.misc import imread, imresize, imshow
import random
import pickle
from shutil import copy2
import time as tm
from sklearn import preprocessing


from skimage.data import imread
from skimage.io import imshow,imsave
from skimage import img_as_float
import pandas as pd
import numpy as np
#import cv2
from skimage.util import crop
from skimage.transform import rotate
from skimage.transform import resize
import matplotlib.pyplot as plt

import math

#######################################################################################################################
#Deep Learning Example with Keras and Lasagne
########################################################################################################################


########################################################################################################################
#
########################################################################################################################
def deg_angle_between(x1,y1,x2,y2):
    from math import atan2, degrees, pi
    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = degrees(rads)
    return(degs)

########################################################################################################################
#
########################################################################################################################
def get_rotated_cropped_fish(img, x1, y1, x2, y2):
    (h, w) = img.shape[:2]
    # calculate center and angle
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    angle = np.floor(-deg_angle_between(x1, y1, x2, y2))
    # print('angle=' +str(angle) + ' ')
    # print('center=' +str(center))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    fish_length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    cropped = rotated[(max((center[1] - fish_length / 1.8), 0)):(max((center[1] + fish_length / 1.8), 0)),
              (max((center[0] - fish_length / 1.8), 0)):(max((center[0] + fish_length / 1.8), 0))]
    #imshow(img)
    #imshow(rotated)
    #imshow(cropped)

    resized = resize(cropped, (224, 224))
    return (resized)

########################################################################################################################
#Read and Load train data from subfolders
########################################################################################################################
def load_train():

    path2=file_path+'train/NoF/'
    path = os.path.join(path2,'*.jpg')
    target_path = os.path.join(file_path,'train/NoF_Processed')

    files = glob.glob(path)
    print('Load folder %s , Total files :- %d' %(format(file_path),len(files)))

    images = list()
    img_filename_list = list()
    for fl in files:
        flbase = os.path.basename(fl)
        img = cv2.imread(fl)
        resized = resize(img, (224, 224))

        #resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        images.append(resized)
        img_filename_list.append(flbase)

    for i in range(len(images)):
        imsave(os.path.join(target_path,str(img_filename_list[i])), images[i])


########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging():

    print("Starting Train feature creation....... at Time: %s" % (tm.strftime("%H:%M:%S")))
    load_train()
    print("Ending Train feature creation....... at Time: %s" % (tm.strftime("%H:%M:%S")))


########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, use_cache, train_folder, test_folder, restore_from_last_checkpoint,\
        img_rows,img_cols,color_type_global,nb_epoch,batch_size

    img_rows, img_cols = 64, 64
    color_type_global = 3
    batch_size = 16
    #nb_epoch = 50
    nb_epoch = 5

    use_cache = 0
    restore_from_last_checkpoint = 0

    train_folder = 'train/annotate'
    test_folder = 'train/cropped_imgs'

    if(platform.system() == "Windows"):

        file_path = 'C:\\Python\\Others\\data\\Nature_Conservancy_Fisheries_Monitoring'

    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Nature_Conservancy_Fisheries_Monitoring/'

    Data_Munging()


########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)