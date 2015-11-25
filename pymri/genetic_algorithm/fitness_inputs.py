"""
1. Train and test the network.
2. Save best weights and biases.
3. Load best weights and biases.
4. Ftiness function: feedforward, max(upper class - lower)
"""


# evaluate individual:
def get_prob_class(net, x, label):
    output_layer = net.feedforward(x)
    x_1 = output_layer[0]
    x_2 = output_layer[1]

    # ### fitness function #01
    # maximize prob_class
    # abs(label-1) swithes between 0 and 1

    # prob_class = output_layer[label] - output_layer[abs(label-1)]
    # fitness_value = prob_class[0]

    # ### fitness function #02

    # fitness_value = x_1 / math.sqrt(math.pow(x_1, 2) + math.pow(x_2, 2))

    # ### fitness function #03

    fitness_value = x_1 - x_2 + (1 - x_1 - x_2)/2

    return fitness_value
