__author__ = 'juanpabloisaza'
import copy
from constants import *


# will remove all features starting with first_char, i.e. x1, x2... or y1, y2...
def remove_feature(first_char, s):

    new_s = copy.deepcopy(s)

    for name in list(s.keys().values):
        if first_char in name:
            new_s.drop(name, axis=0, inplace=True)

    return new_s


# automatic feature generation names, i.e. x1, x2... or y1, y2...
def get_feature_names(first_char, num):

    x_names = []
    for i in range(1, num + 1):
        x_names.append(first_char + str(i))

    return x_names


# returns dictionary containing a matrix with ones or zeros
def get_labels(x_data, w=get_w_rules()):

    d = {}
    for index, name in enumerate(get_feature_names(c, num_c)):
        d[name] = (np.dot(x_data, w[:, index]) > 1) * 1
    return d


def get_error(labels, prediction):

    df_out = abs(prediction-labels)/len(labels)
    return df_out.sum(axis=0)