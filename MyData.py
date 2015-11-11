__author__ = 'juan pablo isaza'
from pandas import *
import matplotlib.pyplot as plt
import colorsys
import math
import constants as cts
import util
from sympy.combinatorics.graycode import GrayCode


# from the w will get the data to plot.
def get_plot_data(w, df):
    x1_line = np.array([min(df[cts.x1]), max(df[cts.x1]), ])
    x2_line = (1 - w[0]*x1_line)/w[1]
    return x1_line, x2_line


# gets a binary string of certain length num_y, with a number.
def get_binary_string(number):
    binary_string = bin(number).replace('0b', '')

    zeros = ''
    for i in range(cts.num_c - len(binary_string)):
        zeros += '0'

    return zeros + binary_string


# plots 2D figure showing data, labels, classifiers and rules used to create the labels.
def plot_all(w_learned, data_df):

    for i in range(len(cts.w_rules[0, :])):

        (x1_line_rule, x2_line_rule) = get_plot_data(cts.w_rules[:, i], data_df)
        (x1_line_learned, x2_line_learned) = get_plot_data(w_learned[:, i], data_df)
        plt.plot(x1_line_rule, x2_line_rule, 'k-',
                 x1_line_learned, x2_line_learned, 'r-')

    n = int(math.pow(2, cts.num_c))

    for i in range(n):
        binary_string = get_binary_string(i)

        condition = True
        for index, name in enumerate(util.get_feature_names('c', cts.num_c)):
            condition &= (data_df[name] == int(binary_string[index]))

        my_filter = data_df[condition]
        plt.plot(my_filter[cts.x1], my_filter[cts.x2], 's', color=colorsys.hsv_to_rgb(*(i*1.0/n, 0.8, 0.8)))

    plt.show()


# generates toy data to explore algorithms.
def generate_toy(num_x):

    # get the x features
    df = DataFrame(np.random.randn(cts.num_examples, num_x), columns=util.get_feature_names(cts.x, num_x))
    df = concat([df, df.apply(lambda row: Series(util.get_labels(row)), axis=1)], axis=1)

    return df


#def map_classes_to_code(num_y, num_c, ):

#     = list(GrayCode(4).generate_gray())





