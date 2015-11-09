import pickle

from pymri.dataset import load_nnadl_dataset


mode = 0


if mode == 0:
    path = '/home/jesmasta/amu/master/nifti/bold/'
    training_data, validation_data, test_data = load_nnadl_dataset(
        path,
        # (('ExeCtrl_0', 'ExeCtrl_5'), ('ExeTool_0', 'ExeTool_5')),
        (('ExeTool_0', 'ExeTool_5'), ('ExeCtrl_0', 'ExeCtrl_5')),
        k_features = 784,
        normalize=True,
        scale_0_1=False,
        sizes=(0.5, 0.25, 0.25)
        )

    pickle.dump(
        [training_data, validation_data, test_data],
        open("/tmp/save.p", "wb")
        )

elif mode == 1:
    training_data, validation_data, test_data = pickle.load(
        open("/tmp/save.p", "rb")
        )

def perform(input_features=784):
    # from pymri.model import fnn
    # net = fnn.Network([input_features, 46, 2])
    # net.SGD(training_data, 100, 11, 2.961, test_data=test_data)

    import fnn2
    net = fnn2.Network([784, 41, 2], cost=fnn2.QuadraticCost)
    net.SGD(
        training_data, 100, 11, 3,
        lmbda = 5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True
        )
    return net

net = perform()
