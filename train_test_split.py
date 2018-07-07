#!/bin/env python

import numpy as np
import pandas as pd
import datetime



#
# some variables
#


ratings_file = "mr_newmovie_names.csv"
log_file = "log_file_split"

seed = 17			# used to select random train and test users between all users
train_test_size = 0.1		# fraction of users used for train and test


now = datetime.datetime.now().isoformat()
now = now[ : now.find('.')]


log = open(log_file + "_" + now, "w")

log.write( "seed: {}\n".format(seed) )
log.write( "train_test_size: {}\n".format(train_test_size) )

#
# import dataset
#

ratings = pd.read_csv(ratings_file)
ratings = ratings.drop(["time"], axis=1)

log.write( "ratings.shape: {}\n".format(ratings.shape) )



#
# read users vector
#

users = ratings['userID'].unique()
n_users = users.shape[0]
log.write( "total n_users: {}\n\n".format(n_users) )




                
#
# select users (np vectors) for train and test
#


        
# set a seed
np.random.seed(seed)
indices = np.random.permutation(users.shape[0])
split = int(users.shape[0]*(1-train_test_size))

log.write( "train_test_size {}\n".format(train_test_size) )
log.write( "split at {}\n".format(split) )

train_ind, test_ind = indices[:split], indices[split:]
users_train , users_test = users[ train_ind ], users[ test_ind ]


log.write( "users_train.shape {}\n".format(users_train.shape) )
log.write( "users_test.shape {}\n".format(users_test.shape) )


np.save("users_train", users_train)
np.save("users_test", users_test)


log.write( "\nusers vectors saved\n\n" )
