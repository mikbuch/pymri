import pickle

from pymri.dataset import load_nnadl_dataset

mode = 0

if mode == 0:
    path = '/home/jesmasta/amu/master/nifti/bold/'
    # path = '/home/jesmasta/downloads/pymvpa-exampledata/'
    training_data, validation_data, test_data = load_nnadl_dataset(
        path,
        sizes=(0.75, 0.25)
        )

    pickle.dump(
        [training_data, validation_data, test_data],
        open("/tmp/save.p", "wb")
        )

elif mode == 1:
    training_data, validation_data, test_data = pickle.load(
        open("/tmp/save.p", "rb")
        )

import fnn
# net = fnn.Network([784, 30, 2])
# net.SGD(training_data, 100, 10, 3.0, test_data=test_data)
net = fnn.Network([784, 46, 2])
net.SGD(training_data, 100, 11, 2.961, test_data=test_data)
