import numpy as np
import scipy as sp
import sys
import pandas as pd # pandas
from bs4 import BeautifulSoup as bs
import os, sys, logging, string, glob
import cssutils as cu
import json
import pickle

########################################################################################################################
#Dato Truly Native                                                                                          #
########################################################################################################################

########################################################################################################################
#Parse HTML files
########################################################################################################################
def parse_page(in_file, urlid):
    """ parameters:
            - in_file: file to read raw_data from
            - url_id: id of each page from file_name """
    page = open(file_path+'input/'+in_file, 'r')

    soup = bs(page)
    doc = {
            "id": urlid,
            #"text":parse_text(soup),
            "title":parse_title(soup ),
            "links":parse_links(soup),
            "images":parse_images(soup),
           }

    return doc

########################################################################################################################
#Parse text
########################################################################################################################
def parse_text(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - textdata: a list of parsed text output by looping over html paragraph tags
        note:
            - could soup.get_text() instead but the output is more noisy """
    #textdata = ['']
    textdata = ""

    for text in soup.find_all('p'):
        try:
            #textdata.append(text.text.encode('ascii','ignore').strip())
            textdata = textdata + (text.text.encode('ascii','ignore').strip())+" ,"
        except Exception:
            continue

    return textdata
    #return list(filter(None,textdata))

########################################################################################################################
#Parse title
########################################################################################################################
def parse_title(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - title: parsed title """

    #title = ['']
    title = ""
    try:
        #title.append(soup.title.string.encode('ascii','ignore').strip())
        title = str(soup.title.string.encode('ascii','ignore').strip())

    except Exception:
        return title

    print(soup.title.string)
    print(str(title))
    sys.exit(0)

    return title
    #return filter(None,title)

########################################################################################################################
#Parse link
########################################################################################################################
def parse_links(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - linkdata: a list of parsed links by looping over html link tags
        note:
            - some bad links in here, this could use more processing """

    #linkdata = ['']
    linkdata = ""
    for link in soup.find_all('a'):
        try:
            #linkdata.append(str(link.get('href').encode('ascii','ignore')))
            linkdata = linkdata + (str(link.get('href').encode('ascii','ignore')))+" ,"
        except Exception:
            continue

    return linkdata

    #return list(filter(None,linkdata))

########################################################################################################################
#Parse images
########################################################################################################################
def parse_images(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - imagesdata: a list of parsed image names by looping over html img tags """
    #imagesdata = ['']
    imagesdata = ""

    for image in soup.findAll("img"):
        try:
            #imagesdata.append("%(src)s"%image)
            imagesdata = imagesdata + ("%(src)s"%image) +" ,"
        except Exception:
            continue

    return imagesdata
    #return filter(None,imagesdata)

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging():

    print("***************Starting Data cleansing***************")

    inFolder = file_path+'input/'
    outputDirectory = file_path+'output'

    cu.log.setLevel(logging.CRITICAL)
    json_array, last_bucket = [], str(0)


    #fIn = glob.glob( inFolder + '*.txt')
    fIn =  os.listdir(inFolder)

    for idx, filename in enumerate(fIn):

        if idx % 10000 == 0:
            print ("Processed %d HTML files" % idx)

        filenameDetails = filename.split("/")
        urlId = filenameDetails[-1].split('_')[0]
        bucket = 'input'

        try:
            doc = parse_page(filename, urlId)

        except Exception as e:
            ferr.write("parse error with reason : "+str(e)+" on page "+urlId+"\n")
            continue

        json_array.append(doc)

        if idx > 0 and filename==fIn[-1]:
            print(filename)
            print(len(json_array))
            print ('SAVING BUCKET %s' % bucket)
            out_file = os.path.join(outputDirectory, 'chunk' + bucket + '.json')

            with open(out_file, mode='w') as feedsjson:
                for entry in json_array:

                    json.dump(entry, feedsjson)
                    feedsjson.write('\n')

            feedsjson.close()
            json_array = []

    print ("Scraping completed .. There may be errors .. check log at errors_in_scraping.log")

    print("***************Ending Data cleansing***************")

    return fIn


########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)
    pd.options.mode.chained_assignment = None  # default='warn'

    global file_path, ferr
    file_path = 'C:/Python/Others/data/Kaggle/Dato_Truly_Native/'
    #file_path = '/home/roshan/Desktop/DS/Others/data/Kaggle/Dato_Truly_Native/'

    ferr = open(file_path+"error/errors_in_scraping.log","w")

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################

    Train_DS =  Data_Munging()


########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################

if __name__ == "__main__":
    main(sys.argv)

