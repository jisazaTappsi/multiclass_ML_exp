__author__ = 'juanpabloisaza'
from pandas import *
import numpy as np
import matplotlib.pyplot as plt

num_x = 2
w_rule = np.random.randn(num_x)*2


def get_w_rule():
    global w_rule
    return w_rule


# from the w will get the data to plot.
def get_plot_data(w, df):
    x1_line = np.array([min(df['x1']), max(df['x1']), ])
    x2_line = (1 - w[0]*x1_line)/w[1]
    return x1_line, x2_line


def plot_all(w_rule, w_learned, data_df):

    (x1_line_rule, x2_line_rule) = get_plot_data(w_rule, data_df)
    (x1_line_learned, x2_line_learned) = get_plot_data(w_learned, data_df)

    plt.plot(x1_line_rule, x2_line_rule, 'k-',
             x1_line_learned, x2_line_learned, 'r-',
             data_df[(data_df.y1 == 1)]['x1'], data_df[(data_df.y1 == 1)]['x2'], 'bs',
             data_df[(data_df.y1 == 0)]['x1'], data_df[(data_df.y1 == 0)]['x2'], 'ro')
    plt.show()


# execute rule to decide label.
def get_labels(x_matrix, w=get_w_rule()):
    return (np.dot(x_matrix, w) > 1) * 1


# automatic feature generation names, i.e. x1, x2... or y1, y2...
def get_feature_names(first_char, num):

    x_names = []
    for i in range(1, num + 1):
        x_names.append(first_char + str(i))

    return x_names


# generates toy data to explore algorithms.
def generate_toy_data(num_x):

    df = DataFrame(np.random.randn(30, num_x), columns=get_feature_names('x', num_x))
    df[get_feature_names('y', 1)[0]] = df.apply(get_labels, axis=1)

    return df


# will train a perceptron and return the weights.
def train_perceptron(the_training_data, learning_rate):

    w = np.random.random(num_x)*2  # init weights

    for i in range(10):
        for index, row in the_training_data.iterrows():
            y_prediction = get_labels(x_matrix=row[:-1], w=w)
            w += learning_rate * (row[get_feature_names('y', 1)[0]] - y_prediction) * row[:-1]

    return w


def get_error(labels, prediction):
    return sum(abs(prediction-labels))/len(labels)


train_data = generate_toy_data(num_x)

w_learned = train_perceptron(train_data[:-1], .5)
train_error = get_error(labels=train_data[get_feature_names('y', 1)[0]].tolist(),
                        prediction=get_labels(x_matrix=train_data.as_matrix(columns=get_feature_names('x', num_x))
                                              , w=w_learned))

print("Train error is = " + str(train_error))
plot_all(w_rule, w_learned, train_data)







