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


file_models = sys.argv[1]




#
# some variables
#


ratings_file = "mr_newmovie_names.csv"
users_train_file = "users_train_0.2.npy"
users_test_file = "users_test_0.2.npy"






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









model_file = open(file_models)
models_name = model_file.readlines()


seqs = pussy.make_vectors(users_train, ratings)


d = {"model_name": [], "left_out" : [], "top_from" : [], "movieID" : [], "count" : [], "coverage":[] }


left_out = [1]
tops = [10]

# print(left_out)


for model in models_name:
    model = model.replace("\n", "")
    restored_model = keras.models.load_model(model)
    for lo in left_out:
        for top in tops:
            predictions = pussy.aggregate_predictions(users_train, seqs, restored_model, lo, top, n_movies)
            unique, counts = np.unique(x, return_counts=True)
            d["model_name"].append(model)
            d["left_out"].append(lo)
            d["top_from"].append(top)
            d["movieID"].append(unique)
            d["count"].append(counts)
            d["coverage"].append(unique.shape[0]/n_movies)


    df = pd.DataFrame(d)
    df.to_csv(path_or_buf = "coverage_{}.csv".format(file_models), sep=',', index=False)
			







