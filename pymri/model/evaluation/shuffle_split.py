from sklearn.cross_validation import ShuffleSplit


ss = ShuffleSplit(250, n_iter=4, test_size=0.25, random_state=0)
