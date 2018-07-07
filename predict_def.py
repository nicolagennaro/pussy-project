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




ratings_file = "../mr_newmovie_names.csv"


model_file = sys.argv[1]

output_file = "predictions_" + model_file.split('_')[2] + ".dat"


users_test_file = "users_test.npy"
users_train_file = "users_train.npy"
top100_laplace_file = "top100_laplace.npy"
top100_num_votes_file = "top100_num_votes.npy"

# left_outs = np.array([1,3,5])			# the model predict on the whole sequence except this
left_outs = np.array([1,3])
# top_from_predictions = np.array([1, 5, 10, 20])	# the top from the softmax layer taken as predictions 
top_from_predictions = np.array([1, 5])



#
# read the csv and the test_users vector
#

out_file = open(output_file, "w")

ratings = pd.read_csv(ratings_file)
# ratings = ratings.drop(["time"], axis=1)
n_movies = len(ratings["movieID"].unique())

# print(ratings.head(4))

users = ratings['userID'].unique()
# print(users)


users_test = np.load(users_test_file)

out_file.write( "users_test.shape {}\n".format(users_test.shape) )
n_users_test = users_test.shape[0]
out_file.write("n_users_test {}\n".format(n_users_test))



#
# train generator, same as training
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
# train generator for the sps, return just the last element of the seq, not the softmax of the last ones
#


def train_generator_sps(users_pool, out=1):
	
	while(True):
		
		global n_movies
		
		user = np.random.choice(users_pool)
		d = ratings[ ratings['userID']==user]
		d.sort_values(['time'])
		print(d)
		user_movies_x = d.iloc[ : (d.shape[0] - out), 1 ]
		user_movies_y = d.iloc[ d.shape[0] - out , 1 ]
		print(user_movies_y)
		
		X_train = np.zeros((1, d.shape[0] - 1, n_movies))
		y_train = np.zeros((1, n_movies))
		
		
		for _ in range((d.shape[0] - out)):
			X_train[ 0, _, user_movies_x.iloc[_] ] = 1
		
		y_train[ 0, user_movies_y ] = 1

		
		yield np.array(X_train), np.array(y_train)



#
# load the model
#


restored_model = keras.models.load_model(model_file)

out_file.write("model restored\n")
restored_model.summary(print_fn = lambda x: out_file.write(x + '\n') )




#
# read the arrays for predictions
#


top100_laplace = np.load("top100_laplace.npy")
top100_num_votes = np.load("top100_num_votes.npy")

out_file.write("top100_laplace.shape {}\n".format(top100_laplace.shape))
out_file.write("top100_num_votes.shape {}\n".format(top100_num_votes.shape))


#
# this function returns "out" movies actually seen by the user and the "top" predicted by 
# the model using the sequence up to "out" movies
#


def predict_user(user, in_model, train_gen, out=1, top=10):

	x, y = next(train_gen(np.array([user]), out=out))
	real_movies = y[0].argsort()[ -out : ]
	# print("y[0, real_movies] {}".format(y[0, real_movies]))
	prediction = in_model.predict(x)
	predicted_movies = prediction[0].argsort()[ -top : ]

	return real_movies, predicted_movies





# for overfitting, see later
predicted_in_num_votes = 0




def print_correct(n_us, zero, total, top, mode):
	out_file.write("\nMode: {}\n".format(mode))
	out_file.write("zeros_correct: {} out of {}\n".format(zero, n_us))
	out_file.write("fraction of totally wrong {}\n".format(zero/n_us))
	out_file.write("number of prediction with one correct at least {}\n".format(n_us - zero) )
	out_file.write("total_correct: {}\n".format(total))
	out_file.write("fraction of at least one correct: {}\n".format(total / n_us))
	out_file.write("fraction of total correct and total predictions: {}\n\n".format(  total / (top*n_us)  ))



n_users_test = 4

out_file.write("n_users_test: {}\n".format(n_users_test))


for i in range(left_outs.shape[0]):
	for j in range(top_from_predictions.shape[0]):
	
		zeros_correct_model = 0
		zeros_correct_laplace = 0
		zeros_correct_num_votes = 0

		total_correct_model = 0
		total_correct_laplace = 0
		total_correct_num_votes = 0
	
		for u in range(n_users_test):
			user = users_test[ u ]
			real, pred = predict_user(user, restored_model, train_generator, out=left_outs[i], top=top_from_predictions[j])


			s = np.sum(np.in1d(real, pred))
			
			if s == 0:
				zeros_correct_model += 1
			else:
				total_correct_model += s


			s = np.sum(np.in1d(real, top100_laplace))
			if s == 0:
				zeros_correct_laplace += 1
			else:
				total_correct_laplace += s


			s = np.sum(np.in1d(real, top100_num_votes))
			if s == 0:
				zeros_correct_num_votes += 1
			else:
				total_correct_num_votes += s
				
		out_file.write("\n\n\nLeft out in prediction: {}, top_from_prediction: {}\n".format(left_outs[i], top_from_predictions[j]))
		print_correct(n_users_test, zeros_correct_model, total_correct_model, top_from_predictions[j], "model")
		print_correct(n_users_test, zeros_correct_laplace, total_correct_laplace, top_from_predictions[j], "laplace")
		print_correct(n_users_test, zeros_correct_num_votes, total_correct_num_votes, top_from_predictions[j], "num_votes")
		
		

sys.exit(1)		
		
# i have to finish this one
# this is for precision and recall and sps @k
# look at the paper for details
#

out_file.write("n_users_test: {}\n".format(n_users_test))	


k = 10
out_file.write("k: {}".format(k))

mean = 0
x2 = 0	

for u in range(n_users_test):
	user = users_test[ u ]
	real, pred = predict_user(user, restored_model, train_generator_sps, out=1, top=10)
	s = np.sum(np.in1d(real, pred))
	
	mean += s


print("mean: {}".format(mean/n_users_test))
print("var:  {}".format())

