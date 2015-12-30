"""
name:
fnn_theano.py

type:
script (module included)
"""

# ### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample


# Activation functions for neurons
def linear(z):
    return z


def ReLU(z):
    return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid


# ### Constants
GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify " \
        + "network3.py\nto set the GPU flag to False."
    try:
        theano.config.device = 'gpu'
    except:
        # it's already set
        pass
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify " \
        + "network3.py to set\nthe GPU flag to True."


def share_data(training_data, test_data):
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(test_data)]


# ### Load the data
def load_data_shared(filename="/tmp/fmri.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, test_data = cPickle.load(f)
    f.close()

    return share_data(training_data, test_data)


# ### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = \
            [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output,
                prev_layer.output_dropout,
                self.mini_batch_size
                )
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        self.best_accuracy = None

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data, lmbda=0.0, verbose=True):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        test_x, test_y = test_data

        # compute number of minibatches for training and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients,
        # and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = \
            self.layers[-1].cost(self) + \
            0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in test mini-batches.

        # mini-batch index
        i = T.lscalar()
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        '''
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        '''
        # Do the actual training
        best_test_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if verbose > 1:
                    if iteration % 1000 == 0:
                        print(
                            "Training mini-batch number {0}".format(iteration)
                            )
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    test_accuracy = np.mean(
                        [test_mb_accuracy(j) for j in xrange(num_test_batches)]
                        )
                    if verbose > 1:
                        print(
                            "Epoch {0}: test accuracy {1:.2%}".format(
                                epoch, test_accuracy
                            )
                        )
                    if test_accuracy >= best_test_accuracy:
                        if verbose > 1:
                            print("This is the best test accuracy to date.")
                        best_test_accuracy = test_accuracy
                        best_iteration = iteration
        if verbose:
            print("Finished training network.")
            print(
                "Best test accuracy of {0:.2%} obtained at iteration {1}"
                .format(best_test_accuracy, best_iteration)
                )
        self.best_accuracy = best_test_accuracy


# ### Define layer types
class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and
        the filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0, scale=np.sqrt(1.0/n_out),
                    size=filter_shape
                    ),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # no dropout in the convolutional layers
        self.output_dropout = self.output


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b
            )
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(
            T.dot(self.inpt_dropout, self.w) + self.b
            )

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(
            T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y]
            )

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


# ### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)

'''
# Examples:

training_data, test_data = load_data_shared()

# Deep learning
mini_batch_size = 10

net = Network([
    ConvPoolLayer(
        image_shape=(mini_batch_size, 1, 28, 28),
        filter_shape=(20, 1, 5, 5),
        poolsize=(2, 2)
        ),
    ConvPoolLayer(
        image_shape=(mini_batch_size, 20, 12, 12),
        filter_shape=(40, 20, 5, 5),
        poolsize=(2, 2)
        ),
    FullyConnectedLayer(n_in=40*4*4, n_out=100),
    SoftmaxLayer(n_in=100, n_out=2)],
    mini_batch_size
    )
net.SGD(training_data, 60, mini_batch_size, 0.1, test_data)

# Shallow architecture
mini_batch_size = 10
net = Network([
    FullyConnectedLayer(n_in=784, n_out=100),
    SoftmaxLayer(n_in=100, n_out=2)],
    mini_batch_size
    )
net.SGD(training_data, 120, mini_batch_size, 3., test_data)

'''
