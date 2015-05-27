__author__ = 'juanpabloisaza'
from pandas import *
import numpy as np
import matplotlib.pyplot as plt
import copy

num_x = 2
num_y = 2
num_examples = 100

# nested array  where each element is a weight vector.
w_rules = np.random.randn(num_x, num_y)*2


def get_w_rules():
    global w_rules
    return w_rules


# from the w will get the data to plot.
def get_plot_data(w, df):
    x1_line = np.array([min(df['x1']), max(df['x1']), ])
    x2_line = (1 - w[0]*x1_line)/w[1]
    return x1_line, x2_line


def plot_all(w_rules, w_learned, data_df):

    (x1_line_rule0, x2_line_rule0) = get_plot_data(w_rules[0], data_df)
    (x1_line_rule1, x2_line_rule1) = get_plot_data(w_rules[1], data_df)

    (x1_line_learned0, x2_line_learned0) = get_plot_data(w_learned[0], data_df)
    (x1_line_learned1, x2_line_learned1) = get_plot_data(w_learned[1], data_df)

    filter0 = data_df[(data_df.y1 == 1)]
    filter00 = data_df[(data_df.y1 == 0)]

    filter1 = filter0[(data_df.y2 == 1)]
    filter2 = filter0[(data_df.y2 == 0)]

    filter3 = filter00[(data_df.y2 == 1)]
    filter4 = filter00[(data_df.y2 == 0)]



    '''
    filter1 = data_df.ix[(data_df['y1'] == 1) & (data_df['y2'] == 1)]
    filter2 = data_df[(data_df.y1 == 0 and data_df.y2 == 1)]
    filter3 = data_df[(data_df.y1 == 1 and data_df.y2 == 0)]
    filter4 = data_df[(data_df.y1 == 0 and data_df.y2 == 0)]
    '''

    plt.plot(x1_line_rule0, x2_line_rule0, 'k-',
             x1_line_rule1, x2_line_rule1, 'k-',
             x1_line_learned0, x2_line_learned0, 'r-',
             x1_line_learned1, x2_line_learned1, 'r-',
             filter1['x1'], filter1['x2'], 'bs',
             filter2['x1'], filter2['x2'], 'ro',
             filter3['x1'], filter3['x2'], 'cs',
             filter4['x1'], filter4['x2'], 'mo',)
    plt.show()


# execute rule to decide label.
def get_row_labels(row, w=get_w_rules()):
    return Series({'y1': (np.dot(row, w[0]) > 1) * 1, 'y2': (np.dot(row, w[1]) > 1) * 1})


# execute rule to decide label.
def get_labels(x_matrix, w=get_w_rules()):
    return DataFrame({'y1': (np.dot(x_matrix, w[0]) > 1) * 1, 'y2': (np.dot(x_matrix, w[1]) > 1) * 1})


# automatic feature generation names, i.e. x1, x2... or y1, y2...
def get_feature_names(first_char, num):

    x_names = []
    for i in range(1, num + 1):
        x_names.append(first_char + str(i))

    return x_names


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
    df = concat([df, df.apply(lambda row: get_row_labels([row['x1'], row['x2']]), axis=1)], axis=1)

    return df


# will train a perceptron and return the weights.
def train_perceptron(the_training_data, learning_rate):

    w = np.random.randn(num_x, num_y)*2  # init weights

    for i in range(10):
        for index, row in the_training_data.iterrows():
            x_row = remove_feature('y', row)
            y_row = remove_feature('x', row)
            y_prediction = get_row_labels(row=x_row, w=w)
            w[0] += learning_rate * (y_row['y1']-y_prediction['y1']) * x_row
            w[1] += learning_rate * (y_row['y2']-y_prediction['y2']) * x_row

    return w


def get_error(labels, prediction):

    df_out = abs(prediction-labels)/len(labels)
    return df_out.sum(axis=0)


train_data = generate_toy_data(num_x)

w_learned = train_perceptron(train_data, .5)
train_error = get_error(labels=train_data.loc[:, get_feature_names('y', num_y)],
                        prediction=get_labels(x_matrix=train_data.as_matrix(columns=get_feature_names('x', num_x))
                                              , w=w_learned))


print("Train error is = " + str(train_error))
plot_all(w_rules, w_learned, train_data)




