#!/bin/env python

import pandas as pd
import numpy as np


urm = pd.read_csv("urm.csv", sep=",")
print(urm.shape)


# take a subset of the all dataframe (only first n rows)
nrows=1000
urm_sub=urm.iloc[0:nrows,:].loc[:,['userID','time','title']]
urm_sub.shape

urm1=urm_sub.copy()

urm1=urm1.drop('title', 1).join(pd.get_dummies(pd.DataFrame(urm1['title'].tolist()).stack()).astype(int).sum(level=0))
print(urm1.head())

print(urm1.shape)

urm1.to_csv(path_or_buf="user_movies_1hot.csv", header=True, sep=",", index=False)