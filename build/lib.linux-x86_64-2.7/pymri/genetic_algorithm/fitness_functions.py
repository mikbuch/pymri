import numpy as np
from sklearn.cross_validation import ShuffleSplit

# from pymri.dataset.nnadl_dataset import load_nifti
import pymri.model.fnn as network


def test_network(X, y, times_tested, hidden, eta, minibatch):
    """
    Compute the fitness function. Run neural network n times (times_tested) -
    (each time with different data split). Then take average of best
    performances for particular runs.

    Parameters
    ----------
    X : ndarray
        Input array.
    y : ndarray
        Target for supervised learning (labels, classes).
    times_tested : int
        Test network for n times (each time witch different datasplit).
    hidden : int
        Number of neurons in hidden layer.
    eta: float
        Learning rate.
    minibatch: int
        Size of the minibatch.

    Returns
    -------
    fitness_value : float
        The mean of best performances in particualr runs divided by number of
        test samples.

    See Also
    --------
    examples/ga/ann_params_optimize.py

    """


    # store best performances from model tested with splitimes_testeded data
    best_performances = np.zeros(shape=(times_tested,))

    rs = ShuffleSplit(X.shape[0], n_iter=times_tested, test_size=.25)

    splits = [split for split in rs]

    for model in range(times_tested):
        # X, X_t, y, y_t = split_data(X, y, train_size)
        print('\nmodel trained %d times' % (model))
        training_data, test_data = \
            zip(X[splits[model][0]], y[splits[model][0]]),\
            zip(X[splits[model][1]], y[splits[model][1]])

        net = network.Network([784, hidden, 2])
        net.SGD(training_data, 10, minibatch, eta, test_data=test_data)

        best_performances[model] = net.best_score

    fitness_value = best_performances.mean()/test_data[0][0].shape[0]

    return fitness_value
