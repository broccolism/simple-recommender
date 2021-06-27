#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

TRAIN_PATH = '../data/ml-1m/ratings.train'
TEST_PATH = '../data/ml-1m/ratings.test'
NAMES = ['userId', 'itemId', 'rating', 'timestamp']

train_data = pd.read_table(TRAIN_PATH,
                           sep='::', header=None, names=NAMES)
train_data.drop(columns=['timestamp'], inplace=True)

N_USER = train_data.userId.max()
N_MOVIE = train_data.itemId.max()

rating = np.zeros((N_USER, N_MOVIE))
for row in train_data.itertuples():
    rating[row[1]-1, row[2]-1] = row[3]

trainset = np.copy(rating)


def adj_c_sim(train_data):
    user_mean = train_data.sum(axis=1)/(train_data != 0).sum(axis=1)
    rating_m_sub = np.where(
        (train_data != 0), train_data - user_mean[:, None], train_data)
    sim = np.zeros((N_MOVIE, N_MOVIE))
    for i in range(N_MOVIE):
        for j in range(i, N_MOVIE):
            num = 0
            dem1 = 0
            dem2 = 0
            set_c_u = np.where((train_data[:, i] != 0) * (train_data[:, j]))[0]
            for k in set_c_u:
                num = num + rating_m_sub[k][i] * rating_m_sub[k][j]
                dem1 = dem1 + rating_m_sub[k][i] ** 2
                dem2 = dem2 + rating_m_sub[k][j] ** 2
                sim[i, j] = num / sqrt(dem1 * dem2 + 10 ** -12)

    return sim


sim = adj_c_sim(trainset)

upp_tr = np.triu(sim, k=1)
upp_tr = upp_tr.T
sim = sim+upp_tr

sim = np.where((sim < 0), 0, sim)

mul = trainset.dot(sim)
div = np.zeros((N_USER, N_MOVIE))
for i in range(N_USER):
    nzi = np.nonzero(trainset[i])
    for j in range(N_MOVIE):
        sm = (sim[j, nzi]).sum()
        div[i, j] = sm

pred = mul / div
np.nan_to_num(pred, copy=False)

test_data = pd.read_table(TEST_PATH,
                          sep='::', header=None, names=NAMES)
test_data.drop(columns=['timestamp'], inplace=True)

rating = np.zeros((N_USER, N_MOVIE))
for row in test_data.itertuples():
    rating[row[1]-1, row[2]-1] = row[3]
test_data = np.copy(rating)

MSE = mean_squared_error(test_data[test_data != 0], pred[test_data != 0])
RMSE = sqrt(MSE)

print("RMSE = ", RMSE)
