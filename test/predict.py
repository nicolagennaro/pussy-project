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



ratings_file = "../user_movies_1hot.csv"
model_file = "model_file"





ratings = pd.read_csv(ratings_file)
ratings = ratings.drop(["time"], axis=1)
n_movies = ratings.shape[1]-1

print(ratings.head(4))

users = ratings['userID'].unique()
print(users)





def train_generator(users_pool, out=1):
        while(True):
            
            global n_movies

            user = np.random.choice(users_pool)
            print("user {}".format(user))
            d = ratings[ratings['userID']==user]
            
            X_train_d = d.iloc[ : (d.shape[0] - out ), 1 : ]
            y_train_d = d.iloc[ d.shape[0] - out : , 1 : ]

            X_train_d = d.iloc[ : (d.shape[0] - out ), 1 : ]
            y_train_d = d.iloc[ d.shape[0] - out : , 1 : ]

            X_train = X_train_d.values.reshape(1, d.shape[0] - out, n_movies )
            y_train = np.zeros(n_movies)
            for _ in range(out):
                y_train = y_train + y_train_d.iloc[ _ ].values.reshape(1, n_movies)

            yield np.array(X_train), np.array(y_train)












restored = keras.models.load_model(model_file)

print(restored.summary())


#users_test = np.load('users_test.npy')
#users_test = np.load('users_train.npy')

#print(users_test)

#users_test = np.arange(0)



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
