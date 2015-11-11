__author__ = 'juan pablo isaza'
import numpy as np

num_x = 2  # number of features.
num_y = 2  # number of classes

ECOC_redundancy = 2  # ECOC (Error	correcting	output	codes) will have a 2 bit redundancy
num_c = 2
num_examples = 100
w_rules = np.random.randn(num_x, num_c)*2
c = 'c'  # coding character.
x = 'x'
x1 = 'x1'
x2 = 'x2'


def get_w_rules():
    global w_rules
    return w_rules