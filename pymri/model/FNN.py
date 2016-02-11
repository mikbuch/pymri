class FNN(object):

    def __init__(
            self, type, input_layer_size, hidden_layer_size,
            output_layer_size, mini_batch_size, epochs, learning_rate,
            verbose=False
            ):
        self.type = type
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.net = None

        self.create_network()

    def create_network(self):
        if 'simple' in self.type:

            if self.verbose:
                print('creating FNN simple')

            from pymri.model.ann.simple import Network
            self.net = Network(
                [
                    self.input_layer_size,
                    self.hidden_layer_size,
                    self.output_layer_size
                ]
                )
        else:

            if self.verbose:
                print('creating FNN theano')

            from pymri.model.ann.theano_script import Network
            from pymri.model.ann.theano_script import FullyConnectedLayer
            from pymri.model.ann.theano_script import SoftmaxLayer

            self.net = Network([
                FullyConnectedLayer(
                    n_in=self.input_layer_size, n_out=self.hidden_layer_size
                    ),
                SoftmaxLayer(
                    n_in=self.hidden_layer_size, n_out=self.hidden_layer_size
                    )
                ],
                self.mini_batch_size
                )

    def train_and_test(self, training_data, test_data):

        if 'simple' in self.type:
            self.net.SGD(
                training_data, self.epochs,
                self.mini_batch_size, self.learning_rate,
                test_data=test_data,
                verbose=self.verbose
                )
        else:
            from pymri.model.ann.theano_script import share_data
            training_data, test_data = share_data(training_data, test_data)
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
