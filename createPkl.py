import csv
import re
import numpy
import gzip
import cPickle
import time
import os
import matplotlib.pylab as plt


hashList = {} # set a commun hashlist
QUINITLE_SIZE = 3400 # default is 340000


# manage the hash dictionary and retrive the id of a given hash
def getIdHash(elem,hashList):
    m = re.search('([^,]+=)', elem )
    if m: 
        if m.group(0) not in hashList:
            hashList[m.group(0)] = len(hashList)
        return hashList[m.group(0)]
    else :
        return -1

# Create a new csv from an old one. 
def reWriteCsvAndLabels(quintile=1):

    # Input is 'train.csv' 
    # Output is 'train'+quintile+'.pkl.gz'
    inputFileTrain='/home/marc/Desktop/miProject/train.csv.gz'
    inputFileLabel='/home/marc/Desktop/miProject/trainLabels.csv.gz'

    # Define begining and end index
    # nbRowsTodo = sum(1 for row in csvFileReader)
    # All file is 1700001 lines. 170/5=34
    startAt = (quintile-1)*QUINITLE_SIZE
    nbRowsTodo = QUINITLE_SIZE

    # open the reader CSV reader
    with gzip.open(inputFileTrain, 'rb') as csvFileReader:
        fileReader  = csv.reader(csvFileReader , delimiter=',', quotechar='|')
        

        # Initialize array and skip some lines
        sample = numpy.ndarray(  (nbRowsTodo,len(fileReader.next())-1), dtype='float32'  )
        for i in range(startAt):
            fileReader.next()

        
        # iterate through columns to retrieve the data
        for rowIter,row in enumerate(fileReader):
            for colIter,elem in enumerate(row):


                # Convert all elems to a matrice
                if elem == '' :
                    sample[rowIter][colIter-1] = -1
                elif elem == 'NO':
                    sample[rowIter][colIter-1] = 0
                elif elem == 'YES':
                    sample[rowIter][colIter-1] = 1
                elif re.search('([^,]+=)', elem ):
                    sample[rowIter][colIter-1] = getIdHash(elem,hashList)
                else:
                    sample[rowIter][colIter-1] = float(elem)


            # Show progress
            if not rowIter % 1000:
                print rowIter

            # For testing purpose, end iteration before end
            if rowIter > nbRowsTodo-2:
                break
    

    # open the reader CSV reader
    with gzip.open(inputFileLabel, 'rb') as csvFileReader:
        fileReader  = csv.reader(csvFileReader , delimiter=',', quotechar='|')
        
        # Initialize array and skip some lines
        labels = numpy.ndarray(  (nbRowsTodo,len(fileReader.next())-1), dtype='float32'  )
        for i in range(startAt):
            fileReader.next()


        # Csv lables to matrix labels
        for rowIter,row in enumerate(fileReader):
            labels[rowIter] = [float(i) for i in row[1:]]


            # Show progress
            if not rowIter % 200:
                print rowIter

            # End before the end of the file
            if rowIter > nbRowsTodo-2:
                break


    saveData( (sample,labels), str(quintile) ,QUINITLE_SIZE ,'raw')
    saveHash(      hashList                  ,QUINITLE_SIZE ,'raw')


def normalizeData():
    
    # Import data in python
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) \
         = loadAllDataPython(QUINITLE_SIZE,'raw')


    # Read a vector and run feature standardization on it.
    for featureIdx in range(len(train_x[0])): # number of features to normalize
        
        # Concatenate vectors to get a feature vector all along the dataset
    	featureVec = 0
        featureVec = train_x[:,featureIdx]
        featureVec = numpy.concatenate( (featureVec,valid_x[:,featureIdx]) )
        featureVec = numpy.concatenate( (featureVec,test_x [:,featureIdx]) )

        # Compute mean and standard deviation of feature vector
        mean = numpy.mean(featureVec)
        std = numpy.std(featureVec)

        # apply feature standardization by 
        # Computing the Zero-mean and Unit-variance on feature vector
        train_x[:,featureIdx] = train_x[:,featureIdx] - mean
        train_x[:,featureIdx] = numpy.divide(train_x[:,featureIdx],std)
        valid_x[:,featureIdx] = valid_x[:,featureIdx] - mean
        valid_x[:,featureIdx] = numpy.divide(valid_x[:,featureIdx],std)
        test_x [:,featureIdx] = test_x [:,featureIdx] - mean
        test_x [:,featureIdx] = numpy.divide(test_x [:,featureIdx],std)


    # Save the normalized data with a file labeled with 'norm'
    for i in range(3):
        tmp_x = train_x[QUINITLE_SIZE*i:QUINITLE_SIZE*(i+1)-1,]
        tmp_y = train_y[QUINITLE_SIZE*i:QUINITLE_SIZE*(i+1)-1,]
        saveData( (tmp_x, tmp_y) ,i+1,QUINITLE_SIZE,'norm')
    saveData( (valid_x, valid_y) ,4,QUINITLE_SIZE,'norm')
    saveData( (test_x , test_y ) ,5,QUINITLE_SIZE,'norm')


    return [(train_x, train_y), (valid_x, valid_y)
        ,(test_x, test_y)]



def loadAllDataPython(size,file_label):
    """
        Load the data_x data_y sample and labels

        :type size: int
        :param size: size of the quintile

        :type file_label: string
        :param file_label: extra label of the data, could be "raw", "norm"...
    """
    

    train_x, train_y = loadData(1,QUINITLE_SIZE,'raw')
    for i in range(2,4):
        tmp_x,tmp_y = loadData(i,QUINITLE_SIZE,'raw')
        train_x = numpy.concatenate( (train_x,tmp_x), axis=0 )
        train_y = numpy.concatenate( (train_y,tmp_y), axis=0 )
    
    valid_x, valid_y = loadData(4,QUINITLE_SIZE,'raw')
    test_x , test_y  = loadData(5,QUINITLE_SIZE,'raw')


    return [(train_x, train_y), (valid_x, valid_y),
            (test_x, test_y)]

def loadData(quintile,size,file_label):
    """
        Load the data_x data_y sample and labels

        :type quintile: int
        :param quintile: quintile is the quintile number [1 to 5]

        :type size: int
        :param size: size of the quintile

        :type file_label: string
        :param file_label: extra label of the data, could be raw, normalized...
    """
    file_name = 'quintile_'+file_label+'_'+str(quintile)
    data_x,data_y = loadPkl(size,file_name)

    return (data_x,data_y)

def loadHash(size,file_label):
    """
        Load the hashfile

        :type size: int
        :param size: size of the quintile

        :type file_label: string
        :param file_label: extra label of the data, could be raw, normalized...
    """
    file_name = 'hashList_'+file_label
    hashList = loadPkl(size,file_name)

    return hashList

def loadPkl(size,file_name):
    """
        Load a cPickle in a gzip file.

        :type size: int
        :param size: size of the quintile

        :type file_name: string
        :param file_name: entire file name
    """

    inputDir = '/home/marc/git/text_classification/data/'+str(size)+'/'
    inputFile = inputDir + file_name+'.pkl.gz' 


    f = gzip.open(inputFile, 'rb')
    data = cPickle.load(f)
    f.close()
    print 'Loaded file <'+str(file_name)+'> '

    return (data)


def saveData(data,quintile,size,file_label):
    """
        This function will save in a gzip format a pickle.

        :type data: tuple
        :param data: (data_x, data_y) numpy.matrix

        :type quintile: int
        :param quintile: quintile is the quintile number [1 to 5]

        :type size: int
        :param size: size of the quintile

        :type file_label: string
        :param file_label: extra label of the data, could be raw, normalized...


    """
    file_name = 'quintile_'+file_label+'_'+str(quintile)
    savePkl(data,size,file_name)

def saveHash(data,size,file_label):
    """
        This function will save in a gzip format a pickle.

        :type data: dictionary
        :param data: (data_x, data_y) numpy.matrix

        :type size: int
        :param size: size of the quintile

        :type file_label: string
        :param file_label: extra label of the data, could be raw, normalized...


    """
    file_name = 'hashList_'+file_label
    savePkl(data,size,file_name)  

def savePkl(data,size,file_name):
    """
        This function will save in a gzip format a pickle.

        :type data: whatEver
        :param data: To save in a file...

        :type file_name: string
        :param file_name: name of the file to save

        :type size: int
        :param size: size of the quintile we are dealing with
    """
    #  Define output file
    outputDir  = '/home/marc/git/text_classification/data/'+str(size)+'/'
    outputFile = outputDir + file_name +'.pkl.gz'
    
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Save with cPickle
    t0 = time.time()
    f = gzip.open(outputFile,'wb')
    cPickle.dump( data ,f)
    f.close()
    print 'saved file <'+str(file_name)+'> in ' + str(time.time() - t0)



if __name__ == '__main__':
    
    # for i in range(5):
    #     reWriteCsvAndLabels(i+1)

    # loadAllDataPython()

    normalizeData()
