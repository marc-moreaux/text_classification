import csv
import re
import numpy
import gzip
import cPickle
import time


hashList = {} # set a commun hashlist
QUINITLE_SIZE = 3400 # default is 340000


if __name__ == '__main__':
	
	for i in range(5)    
	    reWriteCsvAndLabels(i+1)


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
    outputFile='/home/marc/git/text_classification/data/quintile_'+str(QUINITLE_SIZE)

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

    print '... saving'
    t0 = time.time()
    f = gzip.open(outputFile + str(quintile) + '.pkl.gz','wb')
    cPickle.dump( (sample,labels) ,f)
    f.close()
    print 'saved quintile in ' + str(time.time() - t0)
    t0 = time.time()
    f = gzip.open('hashList.pkl.gz','wb')
    cPickle.dump( hashList ,f)
    f.close()
    print 'saved hashList in ' + str(time.time() - t0)


def LoadAllData():
	print '... loading data'

    quintiles_path = '/home/marc/git/text_classification/data/quintile_'+str(QUINITLE_SIZE)


    for i in range(3):
        f = gzip.open(quintiles_path + str(i+1) + ".pkl.gz", 'rb')
        if i == 0:
            train_set_x, train_set_y = cPickle.load(f)
        else :
            tmp_x,tmp_y = cPickle.load(f)
            train_set_x = numpy.concatenate( (train_set_x,tmp_x), axis=0 )
            train_set_y = numpy.concatenate( (train_set_y,tmp_y), axis=0 )

        f.close()
        print 'loaded quintile ' +str(i+1)

    f = gzip.open(quintiles_path + str(4) + ".pkl.gz", 'rb')
    valid_set_x, valid_set_y = cPickle.load(f)
    f.close()
    print 'loaded quintile 4'

    f = gzip.open(quintiles_path + str(5) + ".pkl.gz", 'rb')
    test_set_x, test_set_y = cPickle.load(f)
    f.close()
    print 'loaded quintile 5'



def normalizeData():



