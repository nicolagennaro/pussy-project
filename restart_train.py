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
hidden_neurons = int(params["hidden_neurons"])
n_epochs = int(params["n_epochs"])
predict_every = int(params["predict_every"])

epochs_groups = int(np.ceil(n_epochs/predict_every))


validation_steps_ratio = float(params["validation_steps_ratio"])	# the validation size is n_users_train * this variable
learning_rate = float(params["learning_rate"])
epsilon = float(params["epsilon"])			# parameter for adagrad


model_file = "model_file_20"
train_file_log = "training_loss_values"
log_file = "log_train"
# validation_file = "validation_file"

# model_file = model_file + "_" + file_parameters

new_model_file = "model_file_restored"

log = open(log_file, "w")

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

users_train = np.load(users_train_file)
n_users = users_train.shape[0]
log.write( "n_users: {}\n\n".format(n_users) )


steps_per_epoch = n_users						# a sweep of the 'whole' training set
validation_steps = int(validation_steps_ratio*steps_per_epoch)		# this is used for validation during the training

if validation_steps == 0:
	validation_steps = 1


log.write( "steps_per_epoch {}\n".format(steps_per_epoch) )
log.write( "validation steps {}\n".format(validation_steps) )


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


csv_logger = CSVLogger(train_file_log, append=True, separator=',')
    
 
 
for e in range(n_epochs):
        
        restored_model.fit_generator(pussy.train_generator_softmax(ratings, users_train, n_movies, left_out), epochs=1, steps_per_epoch=steps_per_epoch, validation_data=pussy.train_generator_softmax(ratings, users_train, n_movies, left_out), validation_steps=validation_steps, callbacks=[csv_logger])

        if e % 2 == 0:
                restored_model.save("model_file_{}".format(e))   



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

