"""
1. Train and test the network.
2. Save best weights and biases.
3. Load best weights and biases.
4. Ftiness function: feedforward, max(upper class - lower)
"""



# TODO: generate examples and check distributions
# evaluate individual:
def get_prob_class(net, x, label):
    output_layer = net.feedforward(x)
    # maximize prob_class
    # abs(label-1) swithes between 0 and 1
    prob_class = output_layer[label] - output_layer[abs(label-1)]
    return prob_class[0]
