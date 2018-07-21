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


left_out = int(params["left_out"])		# number of ratings of one user left out during training
kept = int(params["kept"])
hidden_neurons = int(params["hidden_neurons"])
n_epochs = int(params["n_epochs"])
predict_every = int(params["predict_every"])

epochs_groups = int(np.ceil(n_epochs/predict_every))

steps_per_epoch = int(params["steps_per_epoch"])
validation_steps_ratio = float(params["validation_steps_ratio"])	# the validation size is n_users_train * this variable
learning_rate = float(params["learning_rate"])
epsilon = float(params["epsilon"])			# parameter for adagrad


model_file = "model_file"
train_file_log = "training_" + file_parameters
log_file = "log_train_" + file_parameters
# validation_file = "validation_file"

model_file = model_file + "_" + file_parameters


#
# open log file
#

log = open(log_file + "_" + file_parameters, "w")

# log.write( "seed: {}\n".format(seed) )
# log.write( "train_test_size: {}\n".format(train_test_size) )
# log.write( "train_random_state: {}\n".format(train_random_state) )
log.write("\nTrain_start:\n\n")

log.write( "model_file: {}\n".format(model_file) ) 
log.write( "n_epochs: {}\n".format(n_epochs) )
log.write( "left_out: {}\n".format(left_out) )
log.write( "learning rate {}\n".format(learning_rate) )
log.write( "epsilon {}\n".format(epsilon) )
log.write( "validation_steps_ratio {}\n".format(validation_steps_ratio))
log.write( "kept: {}\n".format(kept) ) 
log.write( "steps per epoch: {}\n".format(steps_per_epoch) ) 

#
# read the ratings dataset
#



ratings = pd.read_csv(ratings_file)
n_movies = len(ratings["movieID"].unique())


log.write( "ratings.shape: {}\n".format(ratings.shape) )
log.write( "n_movies: {}\n".format(n_movies) )



                

#
# read users vector
#


log.write("reading users train vector...\n")
users_train = np.load(users_train_file)
users_test = np.load(users_test_file)
n_users = users_train.shape[0]
log.write( "n_users_train: {}\n\n".format(n_users) )




validation_steps = int(validation_steps_ratio*steps_per_epoch)		# this is used for validation during the training

if validation_steps == 0:
	validation_steps = 1


log.write( "steps_per_epoch {}\n".format(steps_per_epoch) )
log.write( "validation steps {}\n".format(validation_steps) )

#
# define model
#


in_out_neurons = n_movies
log.write("defining the model\n")
log.write( "hidden_neurons: {}\n".format(hidden_neurons) )
log.write( "in_out_neurons {}\n".format(in_out_neurons) )


model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False, input_shape=(None, in_out_neurons)))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("softmax"))

opt = keras.optimizers.Adagrad(lr=learning_rate, epsilon=epsilon, decay=0.0)

model.compile(loss="categorical_crossentropy", optimizer=opt) 


model.summary(print_fn = lambda x: log.write(x + '\n') )





#
# train model
#



seqs = pussy.make_vectors(users_train, ratings)






start_time = datetime.datetime.now()
log.write( "start training at {}\n".format(start_time) )


csv_logger = CSVLogger(train_file_log, append=True, separator=',')

for i in range(epochs_groups):
	model.fit_generator(pussy.train_generator(seqs, users_train, n_movies, left_out, kept), epochs=predict_every, steps_per_epoch=steps_per_epoch,
			validation_data=pussy.train_generator(seqs, users_train, n_movies, left_out, kept), validation_steps=validation_steps,
			callbacks=[csv_logger])
	model.save("model_file_{}".format(i) + file_parameters)
	#pussy.evaluate_model(model, users_train, ratings, n_movies).to_csv(path_or_buf="{}_{}_train.csv".format(file_parameters,(1+i)*predict_every), header=True, sep=",", index=False)
	#pussy.evaluate_model(model, users_test, ratings, n_movies).to_csv(path_or_buf="{}_{}_test.csv".format(file_parameters,(1+i)*predict_every), header=True, sep=",", index=False)
	


final_time = datetime.datetime.now()
log.write( "\nfinal time {}\n".format(final_time) )
log.write( "\ntotal training time {}\n\n".format(final_time - start_time) )




#   
# save model
#

model.save(model_file + "last")


log.write( "model saved\n\nfinished\n" )
log.close()

