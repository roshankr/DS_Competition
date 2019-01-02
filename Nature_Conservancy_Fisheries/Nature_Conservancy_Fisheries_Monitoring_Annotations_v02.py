import numpy as np
np.random.seed(2016)
import os
import glob
import cv2
import sys
import pandas as pd
import warnings
import platform
from scipy.misc import imread, imresize, imshow
import time as tm
#from skimage.data import imread
from skimage.io import imsave
import pandas as pd
import numpy as np
from skimage.transform import resize

#######################################################################################################################
#Deep Learning Example with Keras and Lasagne
########################################################################################################################
#testing prediction test *only for test purpose

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

    label_files =  [os.path.join(file_path, "save.ckpt-100000.json_eval.json")]

    data_dirs = [os.path.join(file_path,'test/test_stg1')]

    json_list = []

    for c in range(len(label_files)):
        print("processing for :- "+label_files[c])
        images = list()
        labels_list = list()
        img_filename_list = list()
        labels = pd.read_json(label_files[c])
        for i in range(len(labels)):
                line_dict = {}
                xy_dict = {}
                img_filename = labels.iloc[i, 0]
                annotation_grp = list(labels.iloc[i, 1])
                #print("processing image:- " +str(img_filename))

                if len(annotation_grp) == 0:
                    img = cv2.imread(img_filename)
                    resized = resize(img, (224, 224))
                    images.append(resized)
                    labels_list.append(c)
                    img_filename_list.append(os.path.basename(img_filename).replace(".jpg","_0.jpg"))
                else:
                    for annot in range(len(annotation_grp)):
                        try:
                            annotation = annotation_grp[annot]

                            image = imread(img_filename)

                            new_img = get_rotated_cropped_fish(image, np.floor(annotation['x1']), np.floor(annotation['y1']),
                                                   np.floor(annotation['x2']), np.floor(annotation['y2']))

                            images.append(new_img)
                            labels_list.append(c)
                            img_filename_list.append(os.path.basename(img_filename).replace(".jpg","_")+str(annot)+".jpg")
                        except:
                            pass

        dest_dirs = data_dirs[c]+'_Processed'
        for i in range(len(images)):
            imsave(os.path.join(dest_dirs,str(img_filename_list[i])), images[i])

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