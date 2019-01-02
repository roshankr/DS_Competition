import numpy as np
np.random.seed(2016)
import os
from random import shuffle
import sys
import pandas as pd
import warnings
import platform
import time as tm
import json
from collections import OrderedDict
import pandas as pd
import numpy as np
#import cv2

#######################################################################################################################
#Deep Learning Example with Keras and Lasagne
########################################################################################################################

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

    #label_files =  [os.path.join(file_path,train_folder, "bet_labels.json")]
    #data_dirs = [os.path.join(file_path,'train/BET')]

    json_list = []

    for c in range(len(label_files)):
        print("processing for :- "+label_files[c])
        images = list()
        labels_list = list()
        img_filename_list = list()
        labels = pd.read_json(label_files[c])
        for i in range(len(labels)):
            try:
                line_dict = OrderedDict()
                xy_dict = OrderedDict()
                img_filename = labels.iloc[i, 2]
                l1 = pd.DataFrame((labels[labels.filename == img_filename].annotations).iloc[0])

                xy_dict['x1'] = np.floor(l1.iloc[0, 1])
                xy_dict['y1'] = np.floor(l1.iloc[0, 2])
                xy_dict['x2'] = np.floor(l1.iloc[1, 1])
                xy_dict['y2'] = np.floor(l1.iloc[1, 2])

                line_dict["image_path"] = os.path.join(data_dirs[c], img_filename)
                line_dict["rects"] = [xy_dict]
                json_list.append(line_dict)

            except:
                pass

    shuffle(json_list)
    top80 = json_list[:int(len(json_list)*90/100)]
    bot20 = json_list[int(len(json_list)*90/100):]

    outputfile = os.path.join(file_path, "json_train.json")
    with open(outputfile, 'w') as outfile:
        json.dump(top80, outfile,indent=1)

    outputfile = os.path.join(file_path, "json_test.json")
    with open(outputfile, 'w') as outfile:
        json.dump(bot20, outfile, indent=1)

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
        img_rows,img_cols,color_type_global,nb_epoch,batch_size,outputfile

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
        file_path = '/mnt/hgfs/Python/Others/data/Nature_Conservancy_Fisheries_Monitoring/'

    Data_Munging()


########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)