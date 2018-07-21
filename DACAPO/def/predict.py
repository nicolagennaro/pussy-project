#!/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import CSVLogger
from keras import optimizers

import datetime
import os
import sys



import pussy


file_parameters = sys.argv[1]	# path of file with parameters


params = pussy.load_config(file_parameters)



train_test_size = params["train_test_size"]


#
# some variables
#


ratings_file = "mr_newmovie_names.csv"
users_train_file = "users_train_{}.npy".format(train_test_size)
users_test_file = "users_test_{}.npy".format(train_test_size)



new_model_file = "model_file_restored"




# print( train_file + "_" + now)
# print( log_file + "_" + now)






#
# read the ratings dataset
#



ratings = pd.read_csv(ratings_file)
n_movies = len(ratings["movieID"].unique())





                

#
# read users vector
#

users_train = np.load(users_train_file)
users_test = np.load(users_test_file)








#
# restore the model
#




model_file = sys.argv[2]

restored_model = keras.models.load_model(model_file)








seqs = pussy.make_vectors(users_train, ratings)


d = {"left_out" : [], "top_from" : [], "measure" : [], "mean" : [], "sd" : [] }


left_out = [1,3,5]
tops = [10]

# print(left_out)

for lo in left_out:
        for top in tops:
        
		dd = pussy.predict_users_all(users_train, seqs, restored_model, lo, top, n_movies)
		d = pussy.append_to_dict(d, dd, lo, top)


df = pd.DataFrame(d)

df.to_csv(path_or_buf = "predict_train_{}".format(model_file), sep=',', index=False)










#
# test
#





seqs = pussy.make_vectors(users_test, ratings)


d = {"left_out" : [], "top_from" : [], "measure" : [], "mean" : [], "sd" : [] }

left_out = [1,3,5,10]
tops = [5, 10, 15, 20]

# print(left_out)

for lo in left_out:
        for top in tops:

                dd = pussy.predict_users_all(users_test, seqs, restored_model, lo, top, n_movies)
                d = pussy.append_to_dict(d, dd, lo, top)


df = pd.DataFrame(d)

df.to_csv(path_or_buf = "predict_test_{}".format(model_file), sep=',', index=False)



