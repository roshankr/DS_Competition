#!/usr/bin/python

import platform
import sys
import os
import pickle
import time as tm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from  datetime import datetime, timedelta
import numpy as np
import cv2
import xgboost


def videocapture():
    filename = os.path.join(file_path, 'FlickAnimation.avi')
    cap = cv2.VideoCapture(filename)
    X_test = []

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Our operations on the frame come here

            #frame = frame*(1. / 255)

            X_test.append(frame)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2Luv)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            #cv2.imshow('frame', gray)
            #tm.sleep(6)
            #sys.exit(0)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        else:

            i= 0
            for img in X_test:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imshow('frame'+str(i), gray)
                i = i +1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            #tm.sleep(60)
            #break

    # When everything done, release the capture
    #cap.release()
    #cv2.destroyAllWindows()


if __name__ == "__main__":

    global file_path

    print("video capturing....... at Time: %s" % (tm.strftime("%H:%M:%S")))

    if(platform.system() == "Windows"):

        file_path = 'C:\\Python\\Others\\data\\test'

    else:
        file_path = '/mnt/hgfs/Python/Others/data/test/'


    # try:
    videocapture()


    print("--------------- END train.")
