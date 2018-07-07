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


ratings_file = "../mr_newmovie_names.csv"


n_epochs = 1
left_out = 1				# number of ratings of one user left out during training
steps_per_epoch = 10			# a sweep of the 'whole' training set
validation_steps_ratio = 0.05		# this is used for validation during the training
learning_rate = 0.1
epsilon = 0.01

csv_log = True
save_model = True


users_file = "users_train.npy"
train_file = "training_file"
log_file = "log_train"


model_file = sys.argv[1]



now = datetime.datetime.now().isoformat()
now = now[ : now.find('.')]

# tim = now.split(":")
# tim = v[0][-2:] + "_" + v[1] + "_" + v[2]


new_model_file = "model_file_" + now

log = open(log_file + "_" + now, "w")

log.write("\nTrain_restart:\n\n")
log.write("old_model_file:  {}".format(model_file))
log.write("new_model_file:  {}".format(new_model_file))
log.write( "n_epochs: {}\n".format(n_epochs) )
log.write( "left_out: {}\n".format(left_out) )
log.write( "learning rate: {}\n".format(learning_rate) )
log.write( "epsilon {}\n".format(epsilon) )
log.write( "validation_steps_ratio {}\n".format(validation_steps_ratio))



# print( train_file + "_" + now)
# print( log_file + "_" + now)






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

users_train = np.load(users_file)
n_users = users_train.shape[0]
log.write( "n_users: {}\n\n".format(n_users) )


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
# restore the model
#


restored_model = keras.models.load_model(model_file)

log.write("model restored\n")

restored_model.summary(print_fn = lambda x: log.write(x + '\n') )



#
# train model
#



start_time = datetime.datetime.now()
log.write( "start training at {}\n".format(start_time) )


if csv_log:
    csv_logger = CSVLogger(train_file, append=True, separator=',')
    restored_model.fit_generator(train_generator(users_train, left_out), epochs=n_epochs, steps_per_epoch=steps_per_epoch,
              validation_data=train_generator(users_train), validation_steps=validation_steps,
              callbacks=[csv_logger])

    
else:
    restored_model.fit_generator(train_generator(users_train), epochs=n_epochs, steps_per_epoch=steps_per_epoch,
              validation_data=train_generator(users_train), validation_steps=validation_steps)


final_time = datetime.datetime.now()
log.write( "\nfinal time {}\n".format(final_time) )
log.write( "\ntotal training time {}\n\n".format(final_time - start_time) )




#   
# save model
#


if save_model:
    restored_model.save(new_model_file)


log.write( "model saved\n\nfinished\n" )






log.close()

