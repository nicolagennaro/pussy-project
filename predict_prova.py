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
model_file = "model_file"
users_test_file = "users_test.npy"




#
# read the csv and the test_users vector
#

ratings = pd.read_csv(ratings_file)
ratings = ratings.drop(["time"], axis=1)
n_movies = len(ratings["movieID"].unique())

print(ratings.head(4))

users = ratings['userID'].unique()
print(users)


users_test = np.load(users_test_file)
print("users_test.shape {}".format(users_test.shape))
# print(users_test)



#
# train generator, same as training
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
# load the model
#


restored_model = keras.models.load_model(model_file)

print("restored model:\n")
print(restored_model.summary())
print("\n")



#
# this function returns "out" movies actually seen by the user and the "top" predicted by 
# the model using the sequence up to "out" movies
#


def predict_user(user, in_model, train_gen, out, top):

	x, y = next(train_gen(np.array([user]), out=out))
	real_movies = y[0].argsort()[ -out : ]
	# print("y[0, real_movies] {}".format(y[0, real_movies]))
	prediction = in_model.predict(x)
	predicted_movies = prediction[0].argsort()[ -top : ]

	return real_movies, predicted_movies





#
# some predictions and some coeff to look at
#



left_out = 3
top_from_predictions = 10

zeros_correct = 0
total_correct = 0

for _ in range(int(users_test.shape[0]/10)):
	
	user = users_test[_]
	real, pred = predict_user(user, restored_model, train_generator, left_out, top_from_predictions)
	
	# print(real)
	# print(pred)
	
	# print("intersection")
	# print(np.in1d(pred, real))   # return an array of boolean with the length of the first array
	
	if _ % 10 == 0:
		print(_)
	
	s = np.sum(np.in1d(real, pred))
	if s == 0:
		zeros_correct +=1
	else:
		total_correct += s
		
	# print('sum: {}'.format(np.sum(np.in1d(real, pred))))
	# print("intersection")
	# print(np.intersect1d(real, pred))


print("Left out in prediction: {}, top_from_prediction: {}\n".format(left_out, top_from_predictions))

print("zeros_correct: {} out of {}".format(zeros_correct, users_test.shape[0]))
print("fraction of totally wrong {}".format(zeros_correct/int(users_test.shape[0]/10)))
print("number of prediction with one correct at least {}".format(users_test.shape[0]-zeros_correct) )
print("total_correct: {}".format(total_correct))
print("fraction of at least one correct: {}".format(total_correct/int(users_test.shape[0]/10)))
print("fraction of total correct and total predictions: {}".format(  total_correct/(top_from_predictions*users_test.shape[0])  ))









sys.exit(1)

print("left_out {}".format(left_out))

user = users_test[0]
print("user: {}".format(user))

x, y = next(train_generator(np.array([user]), out=left_out))



print("x.shape {}".format(x.shape))
print("y.shape {}".format(y.shape))


real_movies = y[0].argsort()[ -left_out : ]
print("real_movies {}".format(real_movies))
print("y[0, real_movies]".format(y[0, real_movies]))


prediction = restored_model.predict(x)

print("prediction.shape {}".format(prediction.shape))
predicted_movies = prediction[0].argsort()[ -top_from_predictions : ]
print("predicted movies: {}".format(predicted_movies))





# d = ratings[ ratings['userID']==user]

# print(d['movieID'])

# print(y[0, real_movies])

# print("real_movies {}".format(real_movies))
#top_predicted = y.argsort()[-3:]

# print(top_predicted.shape)
















print("not exec")
user = 2


def prediction3(a,p):
    return(np.array((a==0)*p))


mat=ratings.groupby('userID').sum()

print(mat)

seq, out = next( train_generator(np.array([user]), out=3) )

print("seq")
print(seq.shape)
print(out.shape)
print(out[0, :5])

prediction = restored.predict(seq)
print(prediction.shape)
print(prediction[ 0 , :5])

prediction=prediction.reshape((prediction.shape[1]))

p3 = prediction3(mat.iloc[user,:], prediction)
print("p3")
print(p3)

top=3
#np.argmax(result)
print("result")
print(p3.argsort()[-top:])


print(out)
out=out.reshape((out.shape[1]))

print(out.argsort()[-3:])



a = np.zeros(100)
a[20] = 1
a[78] = 1
a[54] = 0.5

print(a.argsort()[-3:])
#print()
