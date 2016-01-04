class CNN(object):

    def __init__(
            self, type, input, receptive, hidden_conv_layer,
            output_layer_size, mini_batch_size, epochs, learning_rate,
            verbose=False
            ):
        self.type = type
        self.input = input
        self.receptive = receptive
        self.hidden_conv_layer = True
        self.output_layer_size = output_layer_size
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.net = None

        if self.hidden_conv_layer:
            self.hidden_size = (self.input - self.receptive + 1) / 2
            self.fully_connected = (self.hidden_size - self.receptive + 1) / 2
        else:
            self.fully_connected = (self.input - self.receptive + 1) / 2

        self.create_network()

    def create_network(self):

        if self.verbose:
            print('creating CNN theano')

        from pymri.model.ann.theano_script import Network
        from pymri.model.ann.theano_script import ConvPoolLayer
        from pymri.model.ann.theano_script import FullyConnectedLayer
        from pymri.model.ann.theano_script import SoftmaxLayer

        if self.hidden_conv_layer:
            self.net = Network([
                ConvPoolLayer(
                    image_shape=(
                        self.mini_batch_size, 1,
                        self.input, self.input
                        ),
                    filter_shape=(20, 1, self.receptive, self.receptive),
                    poolsize=(2, 2)
                    ),
                ConvPoolLayer(
                    image_shape=(
                        self.mini_batch_size, 20,
                        self.hidden_size, self.hidden_size
                        ),
                    filter_shape=(40, 20, self.receptive, self.receptive),
                    poolsize=(2, 2)
                    ),
                FullyConnectedLayer(
                    n_in=40*self.fully_connected*self.fully_connected,
                    n_out=100
                    ),
                SoftmaxLayer(n_in=100, n_out=self.output_layer_size)
                ],
                self.mini_batch_size
                )
        else:
            self.net = Network([
                ConvPoolLayer(
                    image_shape=(
                        self.mini_batch_size, 1,
                        self.input, self.input
                        ),
                    filter_shape=(20, 1, self.receptive, self.receptive),
                    poolsize=(2, 2)
                    ),
                FullyConnectedLayer(
                    n_in=20*self.fully_connected*self.fully_connected,
                    n_out=100
                    ),
                SoftmaxLayer(n_in=100, n_out=self.output_layer_size)],
                self.mini_batch_size
                )

    def train_and_test(self, training_data, test_data):

        from pymri.model.ann.theano_script import share_data
        training_data, test_data = share_data(training_data, test_data)
        # from pymri.model.ann.theano_script import load_data_shared
        # training_data, test_data = load_data_shared()
        self.net.SGD(
            training_data,
            self.epochs,
            self.mini_batch_size,
            self.learning_rate,
            test_data,
            verbose=self.verbose
            )

    def get_accuracy(self):
        return self.net.best_accuracy
