import requests
import numpy as np
import scipy as sp
import sys
import platform
import pandas as pd
import math
import warnings

########################################################################################################################
#
########################################################################################################################

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path

    if(platform.system() == "Windows"):
        file_path = 'C:/Python/Others/data/Kaggle/Walmart_Recruiting_TTC/'
    else:
        file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Walmart_Recruiting_TTC/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Model1 =  pd.read_csv(file_path+'output/Submission_Roshan_xgb_combined_best.csv',sep=',')
    Model2 =  pd.read_csv(file_path+'output/Submission_Roshan_xgb_orig_9_tmp.csv',sep=',')
    Model3 =  pd.read_csv(file_path+'output/Submission_Roshan_NN61.csv',sep=',')

    Sample_DS     = pd.read_csv(file_path+'sample_submission.csv',sep=',')

    Model1 = Model1.drop(['VisitNumber'], axis = 1)
    Model2 = Model2.drop(['VisitNumber'], axis = 1)
    Model3 = Model3.drop(['VisitNumber'], axis = 1)

    #Submission_Roshan_NN61 has a better LB score (0.84, but not good in ensemeble) ..comign to .64456 with same ensemble as below
    #best one till now ( Submission_Roshan_xgb_combined_best * 0.45 + Submission_Roshan_xgb_orig_9_tmp *0.45 + Submission_Roshan_NN5*0.1) - LB : 0.62782
    pred_Actual = ((Model1*0.45) + (Model2*0.45) + (Model3*0.1)).astype(float)
    pred_Actual = np.array(pred_Actual)

    # Model1_val = np.array(Model1)
    # Model2_val = np.array(Model2)
    # #Model3_val = np.array(Model3['Hazard'])
    #
    # pred_Actual = []
    # for i in range(len(Model1_val)):
    #     x = ((Model1_val[i]*0.6) + (Model2_val[i]*0.4))
    #     pred_Actual.append(x)

    #pred_Actual = np.power((Model1_val*Model2_val*Model3_val), (1/3.0))

    #pred_Actual = np.array(np.floor((Model1['prediction'] + Model2['prediction'] + Model3['prediction']+ Model3['prediction'])/4)).astype(int)

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.VisitNumber.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_xgb_combined_NN.csv', index_label='VisitNumber')

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)
