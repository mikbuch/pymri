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
    prob_class = output_layer[0] - output_layer[1]
    if label == 1:
        return prob_class
    else:
        return 1 - (prob_class + output_layer[1])
