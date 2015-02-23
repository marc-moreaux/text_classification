import csv
import re
import numpy
import gzip
import cPickle
import time
import os



hashList = {} # set a commun hashlist
QUINITLE_SIZE = 3400 # default is 340000
DATA_PATH = os.getcwd()+'/data/'




def csvDataToPkl(q_size,file_label='raw'):

    for quintile in range(1,6):
        sample = loadCsvTrainData(quintile, q_size, 'train')
        labels = loadCsvTrainLabelsData(quintile, q_size, 'trainLabels')
            
        # hashList has been declared as global... sorry
        saveData( (sample,labels), str(quintile) ,q_size ,file_label)

        
    test   = loadCsvTestData(q_size,'test')
    saveData(     (test,0)    ,     6        ,q_size, file_label)
    saveHash(     hashList                   ,q_size ,file_label)

def buildNnormalizeData(q_size):
    """Loads the pickles and normalizes the data
        The nomalisation is feature standardization
    """
    
    # Import data in python
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) \
         = loadAllDataPython(q_size,'raw',True)


    # Read a vector and run feature standardization on it.
    for featureIdx in range(len(train_x[0])): # (0,1, ...145) features 
        
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
    for i in range(4):
        tmp_x = train_x[q_size*i:q_size*(i+1),]
        tmp_y = train_y[q_size*i:q_size*(i+1),]
        saveData( (tmp_x, tmp_y) ,i+1,q_size,'norm')
    saveData( (valid_x , valid_y ) ,5,q_size,'norm')
    saveData( (test_x  , test_y  ) ,6,q_size,'norm')


    return [(train_x, train_y), (valid_x, valid_y)
        ,(test_x, test_y)]

def buildIndexedDataPython(q_size,file_label):
    (train_x,train_y),(test_x) = loadAllDataPython2(q_size,file_label)
    # declare features that are better as indexes (doesn't seem linear)
    indexFeatures = numpy.array( [15,17,18,20,22,23,27] )
    indexFeatures = indexFeatures - 1
    idxSameFeature = numpy.array ( [0,31,61,91,116] )

    train_x_new = train_x
    test_x_new  = test_x

    # declare dicctionaries
    dic = [ dict() for z in range(len(indexFeatures)) ]


    # for each feature_x specified and for each sample
    # increment a dictionary specific to the feature
    # and re-assign an index value to the feature
    for dicNb,idxNb in enumerate(indexFeatures):
        for sample in range(len(train_x)):
            for idemFeature in idxSameFeature:
                if train_x[sample,idxNb+idemFeature] not in dic[dicNb]:
                    dic[dicNb][train_x[sample,idxNb+idemFeature]] = len(dic[dicNb])
                train_x_new[sample,idxNb+idemFeature] = dic[dicNb][train_x[sample,idxNb+idemFeature]]
        for sample in range(len(test_x)):
            for idemFeature in idxSameFeature:
                if test_x[sample,idxNb+idemFeature] not in dic[dicNb]:
                    dic[dicNb][test_x[sample,idxNb+idemFeature]] = len(dic[dicNb])
                test_x_new[sample,idxNb+idemFeature] = dic[dicNb][test_x[sample,idxNb+idemFeature]]

    # rectify indexes so they don't overpass
    for dicNb,idxNb in enumerate(indexFeatures):
        if dicNb == 0:
            maxIdx = 0
        else:
            maxIdx = maxIdx + len(dic[dicNb-1])
            for idemFeature in idxSameFeature:
                train_x[:,idxNb+idemFeature] = train_x[:,idxNb+idemFeature] + maxIdx
            

    for i in range(5):
        tmp_x = train_x_new[q_size*i:q_size*(i+1),]
        tmp_y = train_y[q_size*i:q_size*(i+1),]
        saveData( (tmp_x, tmp_y) ,i+1,q_size,'indexed')
    saveData( (test_x_new  , None  ) ,6,q_size,'indexed')
    savePkl(indexFeatures,q_size,'feature_indexes')


    return (train_x_new,train_y),(test_x_new)

def checkNan(q_size):
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) \
        = loadAllDataPython(QUINITLE_SIZE,'raw')

    print numpy.isnan(train_x).any()
    print numpy.isnan(train_y).any()
    print numpy.isnan(valid_x).any()
    print numpy.isnan(valid_x).any()
    print numpy.isnan(test_x ).any()
    print numpy.isnan(test_y ).any()


    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) \
        = loadAllDataPython(QUINITLE_SIZE,'norm')

    print numpy.isnan(train_x).any()
    print numpy.isnan(train_y).any()
    print numpy.isnan(valid_x).any()
    print numpy.isnan(valid_x).any()
    print numpy.isnan(test_x ).any()
    print numpy.isnan(test_y ).any()



# Load CSV files
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
            print 'Quintile: ',quintile,'-',rowIter

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
            print 'Quintile: ',quintile,'-',rowIter
        
        if rowIter > nbRowsTodo-2: # End before the end of the file
            break

    f.close()

    return labels

def loadCsvTestData(q_size, file_name='test'):
    """ Open the test.csv.gz file and create a matrix out of it
        The matrix is a subset of the original data
        it's the 6'th quintile

        :type file_name: string
        :param file_name: entire file name

        :type return: numpy.ndarray
    """
    # open the reader CSV reader
    inputFileTrain = DATA_PATH + file_name +'.csv.gz'
    f = gzip.open(inputFileTrain, 'rb')
    fileReader  = csv.reader(f , delimiter=',', quotechar='|')

    quintile = 6
    startAt = 0
    nbRowsTodo = 545082 #sum(1 for row in fileReader) default is 545082
    if q_size != 340000:
        nbRowsTodo = q_size
    f.seek(0)
    
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
            print 'Quintile: ',quintile,'-',rowIter

        # For quintile purpose, end iteration before end
        if rowIter > nbRowsTodo-2:
            break

    f.close()

    return sample

def getIdHash(elem,hashList):
    """manage the hash dictionary and retrive the id of a given hash
    """
    m = re.search('([^,]+=)', elem )
    if m: 
        if m.group(0) not in hashList:
            hashList[m.group(0)] = len(hashList)
        return hashList[m.group(0)]
    else :
        return -1


# Load cPickle files
def loadAllDataPython2(q_size,file_label):
    train_x, train_y = loadData(1,q_size,file_label)
    for i in range(2,6):
        tmp_x,tmp_y = loadData(i,q_size,file_label)
        train_x = numpy.concatenate( (train_x,tmp_x), axis=0 )
        train_y = numpy.concatenate( (train_y,tmp_y), axis=0 )

    test_x , test_y  = loadData(6,q_size,file_label)
    


    return (train_x, train_y), (test_x)

def loadAllDataPython(q_size,file_label):
    """
        Load the data_x data_y sample and labels

        :type q_size: int
        :param q_size: size of the quintile

        :type file_label: string
        :param file_label: extra label of the data, could be "raw", "norm"...
    """
    
    train_x, train_y = loadData(1,q_size,file_label)
    for i in range(2,4):
        tmp_x,tmp_y = loadData(i,q_size,file_label)
        train_x = numpy.concatenate( (train_x,tmp_x), axis=0 )
        train_y = numpy.concatenate( (train_y,tmp_y), axis=0 )
    
    tmp_x,tmp_y = loadData(4,q_size,file_label)
    train_x = numpy.concatenate( (train_x,tmp_x), axis=0 )
    train_y = numpy.concatenate( (train_y,tmp_y), axis=0 )

    valid_x, valid_y = loadData(5,q_size,file_label)
    test_x , test_y  = loadData(6,q_size,file_label)
    


    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

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

    print 'loading ...',
    f = gzip.open(inputFile, 'rb')
    data = cPickle.load(f)
    f.close()
    print ' - Loaded <'+str(file_name)+'> '

    return (data)

# Save cPickle files
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
    """ This function will save in a gzip format a pickle.

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
    print 'saving  ...',
    # Save with cPickle
    t0 = time.time()
    f = gzip.open(outputFile,'wb')
    cPickle.dump( data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print ' - saved <'+str(file_name)+'> in ' + str(time.time() - t0)

def saveCsv(q_size,data=None,file_name="submission"):
    """This function will save a submission in a csv file.

        :type data: data_y
        :param data: data to save for submission

        :type q_size: int
        :param q_size: size of the quintile we are dealing with

        :type file_name: string
        :param file_name: name of the file to save
 
   """
    outputDir  = DATA_PATH +str(q_size)+'/'
    if data == None: # load data from pkl if not given
        data = loadPkl(q_size,file_name)

    # open CSV writter
    inputFileTrain = outputDir + file_name +'.csv.gz'
    f = open.fgzip(inputFileTrain, 'wb')
    fileWritter  = csv.writer(f , delimiter=',', quotechar='|')

    # start writting rows
    fileWritter.writerow(['id_label']  + ['pred'])
    data_col = numpy.reshape(data, (-1,1) )
    miter=-1
    for row in data_col:
        miter = miter + 1
        id_label = str(1700000+miter/33+1) +'_y'+ str(miter%33+1) # 17001_y1
        fileWritter.writerow( [id_label]+ [ str(row[0]) ] )
        if not miter % 179877:
            print str(miter/179877)+'% done' 
    f.close()



if __name__ == '__main__':
    
    # csvDataToPkl(QUINITLE_SIZE)
    # normalizeData(QUINITLE_SIZE)
    # saveCsv(q_size=340000,file_name="submission")
    buildIndexedDataPython(3400,'raw')