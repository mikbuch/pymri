import numpy as np
from sklearn.cross_validation import ShuffleSplit

# from pymri.dataset.nnadl_dataset import load_nifti
import pymri.model.fnn as network


def test_network(X, y, hidden, eta, minibatch):

    # path = '/home/jesmasta/amu/master/nifti/bold/'
    # path = '/home/jesmasta/downloads/pymvpa-exampledata/'

    # X, y = load_nifti(path)

    # how many times do we test particular model
    cv = 3

    # store best performances from model tested with splitted data
    best_performances = np.zeros(shape=(cv,))

    rs = ShuffleSplit(X.shape[0], n_iter=cv, test_size=.25)

    splits = [split for split in rs]

    for model in range(cv):
        # X, X_t, y, y_t = split_data(X, y, train_size)
        print('\nmodel trained times: %d' % (model))
        training_data, test_data = \
            zip(X[splits[model][0]], y[splits[model][0]]),\
            zip(X[splits[model][1]], y[splits[model][1]])

        net = network.Network([784, hidden, 2])
        net.SGD(training_data, 10, minibatch, eta, test_data=test_data)

        best_performances[model] = net.best_score

    return best_performances.mean()
