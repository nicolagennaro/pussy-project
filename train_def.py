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



#
# some variables
#


ratings_file = "mr_newmovie_names.csv"
users_train_file = "users_train.npy"


left_out = 1			# number of ratings of one user left out during training
hidden_neurons = 20
n_epochs = 10
validation_steps_ratio = 0.05	# the validation size is n_users_train * this variable
learning_rate = 0.1
epsilon = 0.01			# parameter for adagrad



csv_log = True
save_model = True

model_file = "model_file"
train_file = "training_file"
log_file = "log_train"
# validation_file = "validation_file"




now = datetime.datetime.now().isoformat()
now = now[ : now.find('.') ]

model_file = model_file + "_" + now


#
# open log file
#

log = open(log_file + "_" + now, "w")

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
n_users = users_train.shape[0]
log.write( "n_users_train: {}\n\n".format(n_users) )



steps_per_epoch = n_users						# a sweep of the 'whole' training set
validation_steps = int(validation_steps_ratio*steps_per_epoch)		# this is used for validation during the training

if validation_steps == 0:
	validation_steps = 1


log.write( "steps_per_epoch {}\n".format(steps_per_epoch) )
log.write( "validation steps {}\n".format(validation_steps) )



#
# train generator, extract user from train, put data in np array
# this function feeds the NN with a sequence of observations of one user
#


def train_generator(users_pool, out=1):
	
	while(True):
		
		global n_movies
		
		user = np.random.choice(users_pool)
		d = ratings[ ratings['userID']==user]
		d.sort_values(['time'])
		user_movies_x = d.iloc[ : (d.shape[0] - out), 1 ]
		user_movies_y = d.iloc[ d.shape[0] - out : , 1 ]

		
		X_train = np.zeros((1, d.shape[0] - out, n_movies))
		y_train = np.zeros((1, n_movies))
		
		for _ in range((d.shape[0] - out)):
			X_train[ 0, _, user_movies_x.iloc[_] ] = 1
		
		for _ in range(out):
			y_train[ 0, user_movies_y.iloc[_] ] = 1/out

		
		yield np.array(X_train), np.array(y_train)
		
		
		
		

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




start_time = datetime.datetime.now()
log.write( "start training at {}\n".format(start_time) )


if csv_log:
    csv_logger = CSVLogger(train_file, append=True, separator=',')
    model.fit_generator(train_generator(users_train, left_out), epochs=n_epochs, steps_per_epoch=steps_per_epoch,
              validation_data=train_generator(users_train), validation_steps=validation_steps,
              callbacks=[csv_logger])

    
else:
    model.fit_generator(train_generator(users_train), epochs=n_epochs, steps_per_epoch=steps_per_epoch,
              validation_data=train_generator(users_train), validation_steps=validation_steps)


final_time = datetime.datetime.now()
log.write( "\nfinal time {}\n".format(final_time) )
log.write( "\ntotal training time {}\n\n".format(final_time - start_time) )




#   
# save model
#


if save_model:
    model.save(model_file)


log.write( "model saved\n\nfinished\n" )
log.close()

