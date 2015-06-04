__author__ = 'juanpabloisaza'
import numpy as np

num_x = 2
num_y = 2
coding_redundancy = 2  # will have a 2 bit redundancy
num_c = 2
num_examples = 400
w_rules = np.random.randn(num_x, num_c)*2
c = 'c'  # coding character.
x = 'x'
x1 = 'x1'
x2 = 'x2'


def get_w_rules():
    global w_rules
    return w_rules