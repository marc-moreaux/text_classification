__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy 
import math

import theano
import theano.tensor as T

from createPkl import loadAllDataPython, savePkl, loadData, loadPkl


QUINITLE_SIZE = 340000


def inspect_outputs(i, node, fn):
    for output in fn.outputs:
        if numpy.isnan(output[0]).any():
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            exit()



class LogisticRegression(object):


    def __init__(self, input, n_in, n_out, weights=None):


        if weights == None:
            tmp_weight = numpy.zeros((n_in, n_out), dtype=theano.config.floatX)
            tmp_bias   = numpy.zeros((n_out,),      dtype=theano.config.floatX)
        else :
            tmp_weight = numpy.asarray(weights[0],dtype=theano.config.floatX)
            tmp_bias   = numpy.asarray(weights[1],dtype=theano.config.floatX)


        # initialize with 0 the weights W as a matrix of shape (n_in * n_out)
        self.W = theano.shared(
            value=tmp_weight,
            name='W', borrow=True)

        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=tmp_bias,
            name='b', borrow=True )

        # returns a vector of size n_out 
        self.epss = 0.001

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # Try to return a 0/1 array with learned classes
        self.y_pred = T.nnet.hard_sigmoid( (self.p_y_given_x-0.5)*16 )
        
        self.params = [self.W, self.b] # model parameters

    def negative_log_likelihood(self, y):

        pred = theano.tensor.minimum(
            theano.tensor.maximum(
                self.p_y_given_x,
                self.epss),
            1-self.epss)

        return -T.mean(   y  *T.log(  pred) + (1-y)*T.log(1-pred) )

        

        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type) )

        # check if y is of the correct datatype
        
        pred = theano.tensor.minimum(
            theano.tensor.maximum(
                self.p_y_given_x,
                self.epss),
            1-self.epss)

        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return -T.mean(   y  *T.log(  pred) + \
                        (1-y)*T.log(1-pred) )
            return -T.mean(  T.log(self.p_y_given_x)[y>0]  )
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def loadSharedData(q_size):
    ''' Loads the gzipped pickle dataset
    and places it in shared memory

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) \
        = loadAllDataPython(q_size,file_label='norm')

    
    def shared_dataset(data_x, data_y, borrow=True):

        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        
        return shared_x, T.cast(shared_y, 'int32')

    test_x, test_y   = shared_dataset(test_x, test_y)
    valid_x, valid_y = shared_dataset(valid_x, valid_y)
    train_x, train_y = shared_dataset(train_x, train_y)

    rval = [(train_x, train_y), (valid_x, valid_y),
            (test_x, test_y)]
    return rval

def sgd_optimization_tradeshift(learning_rate=0.13, n_epochs=1000,batch_size=10):
    """ Demonstrate stochastic gradient descent optimization of a log-linear
        model

        This is demonstrated on MNIST.

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
    """
    datasets = loadSharedData(QUINITLE_SIZE)
    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x,  test_y  = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_x.get_value(borrow=True).shape[0]  / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # symbolic input. Matrice (n_in*samples)
    y = T.imatrix('y')  # sybolic labels. Matrice (n_out*samples)

    # construct the logistic regression class
    classifier = LogisticRegression(input=x, n_in=145, n_out=33)

    # the cost we minimize during training is the negative log likelihood
    cost = classifier.negative_log_likelihood(y)


    # print_x = theano.function(
    #         inputs = [index],
    #         outputs = x*x
    #         givens = {x: test_x[index * batch_size: (index + 1) * batch_size]},
    #         mode = detect_nan )

    # x_printed = theano.printing.Print('Bwaaaaa')(x)
    # print_x = theano.function([x],x_printed)


    # compiling a Theano function that computes the mistakes that are made by                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_x[index * batch_size: (index + 1) * batch_size],
            y: test_y[index * batch_size: (index + 1) * batch_size] }  )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_x[index * batch_size: (index + 1) * batch_size],
            y: valid_y[index * batch_size: (index + 1) * batch_size] }  )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        mode=theano.compile.MonitorMode(post_func=inspect_outputs).excluding('local_elemwise_fusion', 'inplace'),
        givens={x: train_x[index * batch_size: (index + 1) * batch_size],
                y: train_y[index * batch_size: (index + 1) * batch_size] } )



    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        
        
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)


            # iteration iter
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                
                this_validation_loss = numpy.mean(validation_losses)

                #  assert numpy.all( print_x(minibatch_index) )
                # print print_x(minibatch_index)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %(
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100. ) )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print( ('     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%' ) %  (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100. ) )

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()

    # print results
    print( ('Optimization complete with best validation score of %f %%,'
            'with test performance %f %%')
            % (best_validation_loss * 100., test_score * 100.) )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

    
    m_weight = classifier.params[0].get_value()
    m_bias   = classifier.params[1].get_value()

    savePkl( (m_weight,m_bias) ,QUINITLE_SIZE,'params')

def computePrediction(q_size):
    """Reproduce the previous neural network and compute predictions

    """

    # load data
    data_x, data_y = loadData(6,q_size,'norm')
    weights = loadPkl(q_size,'params')


    # allocate symbolic variables for the data
    index = T.lscalar()
    x = T.matrix('x')   # symbolic input. Matrice (n_in*samples)
    y = T.imatrix('y')  # sybolic labels. Matrice (n_out*samples)

    shared_x = theano.shared(
        numpy.asarray(data_x,dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
        numpy.asarray(data_y, dtype=theano.config.floatX),borrow=True)

    # construct the logistic regression class
    classifier = LogisticRegression(input=x, n_in=145, n_out=33, weights=weights)


    model_result = theano.function(
        inputs=[index],
        outputs=classifier.p_y_given_x,
        givens={x: shared_x[index : ] } )  

    
    y=model_result(0)

    savePkl( y ,QUINITLE_SIZE,'submition')

    print y





if __name__ == '__main__':
    # load_data()
    # sgd_optimization_tradeshift()
    computePrediction(QUINITLE_SIZE)