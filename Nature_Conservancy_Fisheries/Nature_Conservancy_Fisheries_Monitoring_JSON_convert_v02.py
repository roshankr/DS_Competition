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
import glob

#######################################################################################################################
#Deep Learning Example with Keras and Lasagne
########################################################################################################################

########################################################################################################################
#Read and Load train data from subfolders
########################################################################################################################
def load_train():

    data_dirs = [os.path.join(file_path,'test/test_stg1')]
    #label_files =  [os.path.join(file_path,train_folder, "bet_labels.json")]
    #data_dirs = [os.path.join(file_path,'train/BET')]

    json_list = []
    path = os.path.join(file_path,'test/test_stg1', '*.jpg')
    files = glob.glob(path)

    for fl in files:
            flbase = os.path.basename(fl)
            try:
                line_dict = OrderedDict()
                xy_dict = OrderedDict()
                img_filename = flbase

                xy_dict['x1'] = 1
                xy_dict['y1'] = 1
                xy_dict['x2'] = 1
                xy_dict['y2'] = 1

                line_dict["image_path"] = fl
                line_dict["rects"] = [xy_dict]
                json_list.append(line_dict)

            except:
                pass

    #shuffle(json_list)
    top80 = json_list[:int(len(json_list)*90/100)]
    bot20 = json_list[int(len(json_list)*90/100):]

    outputfile = os.path.join(file_path, "json_eval.json")
    with open(outputfile, 'w') as outfile:
        json.dump(json_list, outfile,indent=1)

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