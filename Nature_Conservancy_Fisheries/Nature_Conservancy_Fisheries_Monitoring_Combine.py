import requests
import numpy as np
import scipy as sp
import sys
import platform
import pandas as pd
import math
import numpy as np
import warnings

########################################################################################################################
#Search Results Relevance                                                                                              #
########################################################################################################################
def clip_df(df, clip, classes):

    # Clip the values
    df = df.clip(lower=(1.0 - clip)/float(classes - 1), upper=clip)

    # Normalize the values to 1
    df = df.div(df.sum(axis=1), axis=0)

    return df
########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path

    if(platform.system() == "Windows"):
        file_path = 'C:/Python/Others/data/Nature_Conservancy_Fisheries_Monitoring/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Nature_Conservancy_Fisheries_Monitoring/'
        #file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Nature_Conservancy_Fisheries_Monitoring/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Model1 =  pd.read_csv(file_path+'subm/50epoch_5aug_0.02clip_2runs_vggbn_inception_resnet.csv',sep=',')
    Model2 =  pd.read_csv(file_path+'subm/submission_NCFM_2017-04-11-21-35.csv',sep=',')
    #Model3 =  pd.read_csv(file_path+'subm/8epoch_10aug_0.01clip_5runs_vggbn_inception_resnet.csv',sep=',')

    Model1_val = Model1.sort(['image'], ascending=[1])
    Model2_val = Model2.sort(['image'], ascending=[1])

    print(Model1_val.head())
    #Model3_val = Model3.sort(['image'], ascending=[1])

    fishes = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
    pred_Actual = pd.DataFrame()
    pred_Actual['image'] = Model1_val['image']

    clip = 0.02
    for fish in fishes:
        test_val = ((Model1_val[fish]*0.7) + (Model2_val[fish]*0.3) )
        #pred_Actual[fish] =np.clip(test_val, clip, 1-clip)

        clip2 = 0.9
        classes = 8
        pred_Actual[fish] = test_val

        pred_Actual[fish] = pred_Actual[fish].clip(lower=(1.0 - clip) / float(classes - 1), upper=clip)

    print(pred_Actual.head())
    # Normalize the values to 1
    #print(pred_Actual[fishes].sum(axis=1))


    pred_Actual[fishes] = pred_Actual[fishes].div(pred_Actual[fishes].sum(axis=1), axis=0)
    print(pred_Actual.head())

    #pred_Actual = np.power((Model1_val*Model2_val*Model3_val), (1/3.0))

    #pred_Actual = np.array(np.floor((Model1['prediction'] + Model2['prediction'] + Model3['prediction']+ Model3['prediction'])/4)).astype(int)

    #Get the predictions for actual data set
    #preds = pd.DataFrame(pred_Actual, index=Sample_DS.Id.values, columns=Sample_DS.columns[1:])
    pred_Actual.to_csv(file_path+'subm/Submission_Roshan_combined2.csv',index=False)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)

