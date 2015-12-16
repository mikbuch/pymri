class FNN(object):

    def __init__(
            self, type, input_layer_size, hidden_layer_size,
            output_layer_size, minibatch_size, epochs, learning_rate
            ):
        self.type = type
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.net = None

        self.create_network()

    def create_network(self):
        if 'simple' in self.type:

            print('creating FNN simple')

            import pymri.model.ann.simple as simple
            self.net = simple.Network(
                [
                    self.input_layer_size,
                    self.hidden_layer_size,
                    self.output_layer_size
                ]
                )
        else:

            print('creating FNN theano')

            import pymri.model.ann.theano_script as theano_script

            mini_batch_size = 10
            self.net = theano_script.Network([
                theano_script.FullyConnectedLayer(n_in=784, n_out=100),
                theano_script.SoftmaxLayer(n_in=100, n_out=2)], self.minibatch_size)

    def train_and_test(self, training_data, test_data):

        if 'simple' in self.type:
            self.net.SGD(
                training_data, self.epochs,
                self.minibatch_size, self.learning_rate,
                test_data=test_data
                )
        else:
            training_data, validation_data, test_data = \
                network3.load_data_shared()
            self.net.SGD(
                training_data, 300, self.mini_batch_size, 0.1,
                validation_data, test_data
                )

    def get_accuracy(self):
        return self.net.best_accuracy
