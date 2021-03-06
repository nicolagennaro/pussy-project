#!/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import CSVLogger

import datetime
import os
import sys

#
# some variables
#


ratings_file = "../mr_newmovie_names.csv"


seed = 17			# used to select random train and test users between all users
train_test_size = 0.1		# fraction of users used for train and test
train_random_state = 3		# seed for training
hidden_neurons = 300
n_epochs = 1
left_out = 1			# number of ratings of one user left out during training


csv_log = True
save_model = True

model_file = "model_file"
train_file = "training_file"
validation_file = "validation_file"



#
# create the folder 
#

now = datetime.datetime.now().isoformat()
now = now[ : now.find('.') ]


try:
    os.makedirs(now)
    os.chdir(now)

except OSError:
    print("cannot create directory")
    sys.exit(1)



#
# open log file
#

log_file = open("log_file", "w")

log_file.write( "seed: {}\n".format(seed) )
log_file.write( "train_test_size: {}\n".format(train_test_size) )
log_file.write( "train_random_state: {}\n".format(train_random_state) )
log_file.write( "hidden_neurons: {}\n".format(hidden_neurons) )
log_file.write( "n_epochs: {}\n".format(n_epochs) )
log_file.write( "left_out: {}\n".format(left_out) )





#
# read the ratings dataset
#



ratings = pd.read_csv(ratings_file)
ratings = ratings.drop(["time"], axis=1)
n_movies = len(ratings["movieID"].unique())


log_file.write( "ratings.shape: {}\n".format(ratings.shape) )
log_file.write( "n_movies: {}\n".format(n_movies) )



                

#
# read users vector
#

users = ratings['userID'].unique()
n_users = users.shape[0]
log_file.write( "n_users: {}\n\n".format(n_users) )




                
#
# select users (np vectors) for train and test
#


        
# set a seed
np.random.seed(seed)
indices = np.random.permutation(users.shape[0])
split = int(users.shape[0]*(1-train_test_size))

log_file.write( "train_test_size {}\n".format(train_test_size) )
log_file.write( "split at {}\n".format(split) )

train_ind, test_ind = indices[:split], indices[split:]
users_train , users_test = users[ train_ind ], users[ test_ind ]


log_file.write( "users_train.shape {}\n".format(users_train.shape) )
log_file.write( "users_test.shape {}\n".format(users_test.shape) )


np.save("users_train", users_train)
np.save("users_test", users_test)


log_file.write( "\nusers vectors saved\n\n" )





#
# train generator, extract user from train, put data in np array
# this function feeds the NN with a sequence of observations of one user
#


def train_generator(users_pool, out=1):
	
	while(True):
		
		global n_movies
		
		user = np.random.choice(users_pool)
		d = ratings[ ratings['userID']==user]
		
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

log_file.write( "in_out_neurons {}\n".format(in_out_neurons) )
log_file.write( "hidden_neurons {}\n".format(hidden_neurons) )


model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False, input_shape=(None, in_out_neurons)))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adagrad") 


model.summary(print_fn = lambda x: log_file.write(x + '\n') )





#
# train model
#


steps_per_epoch = 10		    	# a sweep of the 'whole' training set
validation_steps = 2		# this is used for validation during the training
	
if validation_steps == 0:
	validation_steps = 1


log_file.write( "\nsteps_per_epoch {}\n".format(steps_per_epoch) )
log_file.write( "validation steps {}\n\n".format(validation_steps) )


start_time = datetime.datetime.now()
log_file.write( "start training at {}\n".format(start_time) )


if csv_log:
    csv_logger = CSVLogger(train_file, append=True, separator=',')
    model.fit_generator(train_generator(users_train, left_out), epochs=n_epochs, steps_per_epoch=steps_per_epoch,
              validation_data=train_generator(users_train), validation_steps=validation_steps,
              callbacks=[csv_logger])

    
else:
    model.fit_generator(train_generator(users_train), epochs=n_epochs, steps_per_epoch=steps_per_epoch,
              validation_data=train_generator(users_train), validation_steps=validation_steps)


final_time = datetime.datetime.now()
log_file.write( "\nfinal time {}\n".format(final_time) )
log_file.write( "\ntotal training time {}\n\n".format(final_time - start_time) )




#   
# save model
#


if save_model:
    model.save(model_file)


log_file.write( "model saved\n\nfinished\n" )
log_file.close()

