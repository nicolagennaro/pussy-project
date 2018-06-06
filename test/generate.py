#!/bin/env python

import numpy as np
import pandas as pd
# import tensorflow as tf
# import keras


import datetime
#from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split




def create_fake_dataset(size=100):

    flow = (list(range(1,10,1)) + list(range(10,1,-1))) * size
    uID = np.random.choice(np.arange(start=1, stop=800), size=600, replace=False)
    uID = np.sort(uID)
    print(len(flow))
    print("uid.shape {}".format(uID.shape))

    count = 0
    users = []
    
    for i in range(uID.shape[0]-1):
        seq_len = np.random.randint(20,30)
        u = [ uID[i] for _ in range(seq_len) ]
        users = users + u
        count = count + seq_len

    print("count : {}".format(count))
    u = [ uID[-1] for _ in range(len(flow)-count) ]
    users = users + u
    print("len users {}".format(len(users)))
    data = pd.DataFrame({'userID': users, 'movieID': flow})
    one_hot = pd.get_dummies(data['movieID'])
    data = data.drop('movieID', axis=1)
    data = data.join(one_hot)
    #data = data.sort_values("userID", ascending=True)

    return data
    


ratings = create_fake_dataset(size=1000)

print(ratings.shape)
print(ratings.head(3))



ratings.to_csv(path_or_buf="ratings.csv", sep=",", header=True, index=False)
