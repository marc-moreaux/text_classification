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
DATA_PATH = '/home/marc/git/text_classification/data/'



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


def csvDataToPkl(q_size,file_label='row'):

    for quintile in range(1,6):
        sample = loadCsvTrainData(quintile, q_size, 'train')
        labels = loadCsvTrainLabelsData(quintile, q_size, 'trainLabels')
    
        saveData( (sample,labels), str(quintile) ,q_size ,'raw')
        saveHash(     hashList                   ,q_size ,'raw') # hashList has been declared as global... sry


def loadCsvTrainData(quintile, q_size, file_name='train'):
    """ Open the train.csv.gz file and create a matrix out of it
        The matrix is a subset of the original data
        it's the x'th quintile

        :type quintile: int
        :param quintile: quintile number

        :type q_size: int
        :param q_size: size of the quintile

        :type file_name: string
        :param file_name: entire file name

        :type return: numpy.ndarray
    """


    # open the reader CSV reader
    inputFileTrain = DATA_PATH + file_name +'.csv.gz'
    f = gzip.open(inputFileTrain, 'rb')
    fileReader  = csv.reader(f , delimiter=',', quotechar='|')

    # Define matrices indexes
    startAt = (quintile-1)*q_size
    nbRowsTodo = q_size
    
    # Initialize array and skip non-desired lines
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

        # For quintile purpose, end iteration before end
        if rowIter > nbRowsTodo-2:
            break

    f.close()

    return sample

def loadCsvTrainLabelsData(quintile, q_size, file_name='trainLabels'):
    """ Open the trainLabels.csv.gz file and create a matrix out of it
        The matrix is a subset of the original data
        it's the x'th quintile

        :type quintile: int
        :param quintile: quintile number

        :type q_size: int
        :param q_size: size of the quintile

        :type file_name: string
        :param file_name: entire file name

        :type return: numpy.ndarray
    """


    # open the reader CSV reader
    inputFileLabel = DATA_PATH + file_name +'.csv.gz'
    print inputFileLabel
    f = gzip.open(inputFileLabel, 'rb')
    fileReader  = csv.reader(f , delimiter=',', quotechar='|')
    
    # Define matrices indexes
    startAt = (quintile-1)*q_size
    nbRowsTodo = q_size

    # Initialize array and skip some lines
    labels = numpy.ndarray(  (nbRowsTodo,len(fileReader.next())-1), dtype='float32'  )
    for i in range(startAt):
        fileReader.next()


    # Csv lables to matrix labels
    for rowIter,row in enumerate(fileReader):
        labels[rowIter] = [float(i) for i in row[1:]]

        if not rowIter % 200: # Show progress
            print rowIter
        
        if rowIter > nbRowsTodo-2: # End before the end of the file
            break

    f.close()

    return labels


def loadAllDataPython(q_size,file_label):
    """
        Load the data_x data_y sample and labels

        :type q_size: int
        :param q_size: size of the quintile

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

def loadData(quintile,q_size,file_label):
    """
        Load the data_x data_y sample and labels

        :type quintile: int
        :param quintile: quintile is the quintile number [1 to 5]

        :type q_size: int
        :param q_size: size of the quintile

        :type file_label: string
        :param file_label: extra label of the data, could be raw, normalized...
    """
    file_name = 'quintile_'+file_label+'_'+str(quintile)
    data_x,data_y = loadPkl(q_size,file_name)

    return (data_x,data_y)

def loadHash(q_size,file_label):
    """
        Load the hashfile

        :type q_size: int
        :param q_size: size of the quintile

        :type file_label: string
        :param file_label: extra label of the data, could be raw, normalized...
    """
    file_name = 'hashList_'+file_label
    hashList = loadPkl(q_size,file_name)

    return hashList

def loadPkl(q_size,file_name):
    """
        Load a cPickle in a gzip file.

        :type q_size: int
        :param q_size: size of the quintile

        :type file_name: string
        :param file_name: entire file name
    """

    inputDir = DATA_PATH + str(q_size)+'/'
    inputFile = inputDir + file_name+'.pkl.gz' 


    f = gzip.open(inputFile, 'rb')
    data = cPickle.load(f)
    f.close()
    print 'Loaded file <'+str(file_name)+'> '

    return (data)


def saveData(data,quintile,q_size,file_label):
    """
        This function will save in a gzip format a pickle.

        :type data: tuple
        :param data: (data_x, data_y) numpy.matrix

        :type quintile: int
        :param quintile: quintile is the quintile number [1 to 5]

        :type q_size: int
        :param q_size: size of the quintile

        :type file_label: string
        :param file_label: extra label of the data, could be raw, normalized...


    """
    file_name = 'quintile_'+file_label+'_'+str(quintile)
    savePkl(data,q_size,file_name)

def saveHash(data,q_size,file_label):
    """
        This function will save in a gzip format a pickle.

        :type data: dictionary
        :param data: (data_x, data_y) numpy.matrix

        :type q_size: int
        :param q_size: size of the quintile

        :type file_label: string
        :param file_label: extra label of the data, could be raw, normalized...


    """
    file_name = 'hashList_'+file_label
    savePkl(data,q_size,file_name)  

def savePkl(data,q_size,file_name):
    """
        This function will save in a gzip format a pickle.

        :type data: whatEver
        :param data: To save in a file...

        :type file_name: string
        :param file_name: name of the file to save

        :type q_size: int
        :param q_size: size of the quintile we are dealing with
    """
    #  Define output file
    outputDir  = DATA_PATH +str(q_size)+'/'
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

    # normalizeData()

    csvDataToPkl(3400)