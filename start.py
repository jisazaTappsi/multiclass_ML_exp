__author__ = 'juanpabloisaza'
from util import *
import constants as cts
import MyData
import perceptron
from pandas import *


train_data = MyData.generate_toy(cts.num_x)

w_learned = perceptron.train(train_data, .8)

prediction = DataFrame(get_labels(x_data=train_data.as_matrix(columns=get_feature_names(cts.x, cts.num_x))
                                  , w=w_learned))
train_error = get_error(labels=train_data.loc[:, get_feature_names(cts.c, cts.num_c)],
                        prediction=prediction)


print("Train error is = " + str(train_error))
MyData.plot_all(w_learned, train_data)