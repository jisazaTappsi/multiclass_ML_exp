__author__ = 'juanpabloisaza'

import unittest
from util import *
from pandas import *


class ErrorTest(unittest.TestCase):

    def test_zero_error(self):

        labels = DataFrame(data={'c1': [0, 0], 'c2': [0, 0]})
        prediction = DataFrame(data={'c1': [0, 0], 'c2': [0, 0]})

        error = get_error(labels, prediction)
        self.assertEqual(error['c1'], 0.0)
        self.assertEqual(error['c2'], 0.0)

    def test_one_error(self):

        labels = DataFrame(data={'c1': [1, 1], 'c2': [1, 1]})
        prediction = DataFrame(data={'c1': [0, 0], 'c2': [0, 0]})

        error = get_error(labels, prediction)
        self.assertEqual(error['c1'], 1.0)
        self.assertEqual(error['c2'], 1.0)

    def test_half_error(self):

        labels = DataFrame(data={'c1': [0, 0], 'c2': [0, 0]})
        prediction = DataFrame(data={'c1': [0, 0], 'c2': [0, 0]})

        error = get_error(labels, prediction)
        self.assertEqual(error['c1'], 1.0)
        self.assertEqual(error['c2'], 1.0)