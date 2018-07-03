#!/bin/env python

import numpy as np
import pandas as pd


ratings_file = "mr_newmovie_names.csv"


seed = 17			# used to select random train and test users between all users
train_test_size = 0.1		# fraction of users used for train and test
train_random_state = 3		# seed for training
hidden_neurons = 300
n_epochs = 1
left_out = 1			# number of ratings of one user left out during training





ratings = pd.read_csv(ratings_file)
# ratings = ratings.drop(["time"], axis=1)
print("ratings")
print(ratings.head())

print(ratings["movieID"].unique())
n_movies = len(ratings["movieID"].unique())
print("n_movies {}".format(n_movies))




users = ratings['userID'].unique()
n_users = users.shape[0]
print("users")
print(users)

print("N-users {}".format(n_users))



# set a seed
np.random.seed(seed)
indices = np.random.permutation(users.shape[0])
split = int(users.shape[0]*(1-train_test_size))

train_ind, test_ind = indices[:split], indices[split:]
users_train , users_test = users[ train_ind ], users[ test_ind ]

np.save("users_train", users_train)
np.save("users_test", users_test)



                                                                                    

def train_generator_old(users_pool, out=1):
        while(True):
            
            global n_movies

            user = np.random.choice(users_pool)
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





def train_generator(users_pool, out=1):
	
	while(True):
		
		global n_movies
		
		user = np.random.choice(users_pool)
		print( "selected user {}".format(user) )
		d = ratings[ ratings['userID']==user]
		print("d.shape {}".format(d.shape))
		print(d)
		
		user_movies_x = d.iloc[ : (d.shape[0] - out), 1 ] #.values()
		print( "user_movies_x" )
		print(type(user_movies_x))
		print(user_movies_x)
		
		user_movies_y = d.iloc[ d.shape[0] - out : , 1 ] #.values()
		print( "user_movies_y" )
		print(user_movies_y)
		
		X_train = np.zeros((1, d.shape[0] - out, n_movies))
		y_train = np.zeros(n_movies)
		
		for _ in range((d.shape[0] - out)):
			X_train[ 0, _, user_movies_x.iloc[_] ] = 1
		
		for _ in range(out):
			y_train[ user_movies_y.iloc[_] ] = 1/out
			
			
		
		print("X_train.shape {}". format(X_train.shape))
		print(X_train)
		print("y_train.shape {}". format(y_train.shape))
		print(y_train)
		
		
		
		yield np.array(X_train), np.array(y_train)
		
		
		
x, y = next(train_generator(users, 1))


print("outside")
print(type(x))
print(type(y))

maxx = y.argsort()[-1:]
print("argsort: ")
print(y[maxx], y[maxx+1])


x, y = next(train_generator(users, 1))
x, y = next(train_generator(users, 3))

maxx = y.argsort()[-4:]
print(y[maxx])
# print()

print(x.shape)
print(y.shape)

x, y = next(train_generator_old(users, 1))
print(x.shape)
print(y.shape)
