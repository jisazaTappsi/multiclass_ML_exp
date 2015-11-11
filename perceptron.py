__author__ = 'juan pablo isaza'
from util import *
from constants import *
from pandas import *


# will train a perceptron and return the weights.
def train(the_training_data, learning_rate):

    w = np.random.randn(num_x, num_c)*2  # init weights

    for i in range(5):
        for idx1, row in the_training_data.iterrows():
            x_row = remove_columns_starting_with(c, row)
            c_row = remove_columns_starting_with(x, row)
            c_prediction = Series(get_labels(x_data=x_row, w=w))

            for idx2, name in enumerate(get_feature_names(c, num_c)):
                w[:, idx2] += learning_rate * (c_row[name] - c_prediction[name]) * x_row
    return w


