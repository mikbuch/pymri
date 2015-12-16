class SVC(object):

    def __init__(
            self, C, kernel, gamma, degree
            ):
        self.type = 'svc'
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

        self.accuracy = None

    def train_and_test(self, training_data, test_data):

        from sklearn import svm

        svc = svm.SVC(
            C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree
            )
        # fit samples to targets
        svc.fit(training_data[0], training_data[1])
        y_predicted = svc.predict(test_data[0])

        from sklearn.metrics import accuracy_score
        self.accuracy = accuracy_score(test_data[1], y_predicted)

    def get_accuracy(self):
        return self.accuracy
