__author__ = 'juanpabloisaza'
from pandas import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import colorsys
import math

num_x = 2
num_y = 2
num_examples = 400
w_rules = np.random.randn(num_x, num_y)*2


# automatic feature generation names, i.e. x1, x2... or y1, y2...
def get_feature_names(first_char, num):

    x_names = []
    for i in range(1, num + 1):
        x_names.append(first_char + str(i))

    return x_names


def get_w_rules():
    global w_rules
    return w_rules


# from the w will get the data to plot.
def get_plot_data(w, df):
    x1_line = np.array([min(df['x1']), max(df['x1']), ])
    x2_line = (1 - w[0]*x1_line)/w[1]
    return x1_line, x2_line


# gets a binary string of certain length num_y, with a number.
def get_binary_string(number):
    binary_string = bin(number).replace('0b', '')

    zeros = ''
    for i in range(num_y - len(binary_string)):
        zeros += '0'

    return zeros + binary_string


def plot_all(w_rules, w_learned, data_df):

    for i in range(len(w_rules[0, :])):

        (x1_line_rule, x2_line_rule) = get_plot_data(w_rules[:, i], data_df)
        (x1_line_learned, x2_line_learned) = get_plot_data(w_learned[:, i], data_df)
        plt.plot(x1_line_rule, x2_line_rule, 'k-',
                 x1_line_learned, x2_line_learned, 'r-')

    n = int(math.pow(2, num_y))

    for i in range(n):
        binary_string = get_binary_string(i)

        condition = True
        for index, name in enumerate(get_feature_names('y', num_y)):
            condition &= (data_df[name] == int(binary_string[index]))

        my_filter = data_df[condition]
        plt.plot(my_filter['x1'], my_filter['x2'], 's', color=colorsys.hsv_to_rgb(*(i*1.0/n, 0.8, 0.8)))

    plt.show()


# returns dictionary containing a matrix with ones or zeros
def get_labels(x, w=get_w_rules()):

    d = {}
    for index, name in enumerate(get_feature_names('y', num_y)):
        d[name] = (np.dot(x, w[:, index]) > 1) * 1
    return d


# will remove all features starting with first_char, i.e. x1, x2... or y1, y2...
def remove_feature(first_char, s):

    new_s = copy.deepcopy(s)

    for name in list(s.keys().values):
        if first_char in name:
            new_s.drop(name, axis=0, inplace=True)

    return new_s


# generates toy data to explore algorithms.
def generate_toy_data(num_x):

    # get the x features
    df = DataFrame(np.random.randn(num_examples, num_x), columns=get_feature_names('x', num_x))
    df = concat([df, df.apply(lambda row: Series(get_labels(row)), axis=1)], axis=1)

    return df


# will train a perceptron and return the weights.
def train_perceptron(the_training_data, learning_rate):

    w = np.random.randn(num_x, num_y)*2  # init weights

    for i in range(10):
        for idx1, row in the_training_data.iterrows():
            x_row = remove_feature('y', row)
            y_row = remove_feature('x', row)
            y_prediction = Series(get_labels(x=x_row, w=w))

            for idx2, name in enumerate(get_feature_names('y', num_y)):
                w[:, idx2] += learning_rate * (y_row[name] - y_prediction[name]) * x_row
    return w


def get_error(labels, prediction):

    df_out = abs(prediction-labels)/len(labels)
    return df_out.sum(axis=0)


train_data = generate_toy_data(num_x)

w_learned = train_perceptron(train_data, .8)

prediction = DataFrame(get_labels(x=train_data.as_matrix(columns=get_feature_names('x', num_x))
                                  , w=w_learned))
train_error = get_error(labels=train_data.loc[:, get_feature_names('y', num_y)],
                        prediction=prediction)


print("Train error is = " + str(train_error))
plot_all(w_rules, w_learned, train_data)
