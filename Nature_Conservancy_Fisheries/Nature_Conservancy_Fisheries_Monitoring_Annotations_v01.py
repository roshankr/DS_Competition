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
from math import atan2, degrees, pi
import math

#######################################################################################################################
#Deep Learning Example with Keras and Lasagne
########################################################################################################################


########################################################################################################################
#
########################################################################################################################
def deg_angle_between(x1,y1,x2,y2):
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
    #calculate center and angle
    center = ( (x1+x2) / 2,(y1+y2) / 2)
    angle = np.floor(-deg_angle_between(x1,y1,x2,y2))
    #print('angle=' +str(angle) + ' ')
    #print('center=' +str(center))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    fish_length = np.sqrt((x1-x2)**2+(y1-y2)**2)
    cropped = rotated[np.int((max((center[1]-fish_length/1.8),0))):np.int((max((center[1]+fish_length/1.8),0))) ,
                      np.int(max((center[0]- fish_length/1.8),0)):np.int(max((center[0]+fish_length/1.8),0))]
    #imshow(img)
    #imshow(rotated)
    #imshow(cropped)

    #resized = resize(cropped, (224, 224))
    return (cropped)

########################################################################################################################
#Read and Load train data from subfolders
########################################################################################################################
def load_train():

    label_files =  [os.path.join(file_path,train_folder, "bet_labels.json"),
                    os.path.join(file_path,train_folder, "alb_labels.json"),
                    os.path.join(file_path,train_folder, "yft_labels.json"),
                    os.path.join(file_path,train_folder, "dol_labels.json"),
                    os.path.join(file_path,train_folder, "shark_labels.json"),
                    os.path.join(file_path,train_folder, "lag_labels.json"),
                    os.path.join(file_path,train_folder, "other_labels.json")]

    data_dirs = [os.path.join(file_path,'train/BET'),
                 os.path.join(file_path,'train/ALB'),
                 os.path.join(file_path,'train/YFT'),
                 os.path.join(file_path,'train/DOL'),
                 os.path.join(file_path,'train/SHARK'),
                 os.path.join(file_path,'train/LAG'),
                 os.path.join(file_path,'train/OTHER')]

    # label_files =  [os.path.join(file_path,train_folder, "bet_labels.json")]
    #
    # data_dirs = [os.path.join(file_path,'train/BET')]

    json_list = []

    for c in range(len(label_files)):
        print("processing for :- "+label_files[c])
        images = list()
        labels_list = list()
        img_filename_list = list()
        labels = pd.read_json(label_files[c])
        for i in range(len(labels)):
            #try:
                line_dict = {}
                xy_dict = {}
                img_filename = labels.iloc[i, 2]
                l1 = pd.DataFrame((labels[labels.filename == img_filename].annotations).iloc[0])
                image = imread(os.path.join(data_dirs[c], img_filename))
                images.append(get_rotated_cropped_fish(image,np.floor(l1.iloc[0,1]),np.floor(l1.iloc[0,2]),np.floor(l1.iloc[1,1]),np.floor(l1.iloc[1,2])))

                #print(img_filename)
                #print('success')
                labels_list.append(c)
                img_filename_list.append(img_filename)

                xy_dict['x1'] = np.floor(l1.iloc[0, 1])
                xy_dict['y1'] = np.floor(l1.iloc[0, 2])
                xy_dict['x2'] = np.floor(l1.iloc[1, 1])
                xy_dict['y2'] = np.floor(l1.iloc[1, 2])

                line_dict['image'] = os.path.join(data_dirs[c], img_filename)
                line_dict['rect'] = xy_dict
                json_list.append(line_dict)

            #except:
            #    pass

        dest_dirs = data_dirs[c]+'_Processed'
        for i in range(len(images)):
            imsave(os.path.join(dest_dirs,str(img_filename_list[i])), images[i])

        # for i in range(50):
        #    fig,ax = plt.subplots(nrows=1,ncols=8,sharex="col",sharey="row",figsize=(24,3))
        #    fig.suptitle(str(labels_list[(i*8):(8+i*8)]),fontsize=16)
        #    for j in range(8):
        #        ax[j].imshow(images[j+i*5])

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

    train_folder = 'train/annotate'

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