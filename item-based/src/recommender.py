#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from math import sqrt
from numpy import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

NAMES = ['userId', 'itemId', 'rating', 'timestamp']
N_USER = -1
N_MOVIE = -1


def init_data():
    global N_USER, N_MOVIE
    train_data = pd.read_table('../data/ml-1m/ratings.dat',
                               sep='::', header=None, names=NAMES)
    train_data.drop(columns=['timestamp'], inplace=True)

    N_USER = train_data.userId.max()
    N_MOVIE = train_data.itemId.max()

    rating = np.zeros((N_USER, N_MOVIE))
    for row in train_data.itertuples():
        rating[row[1] - 1, row[2] - 1] = row[3]

    train_matrix = np.copy(rating)
    test_matrix = np.zeros((N_USER, N_MOVIE))

    u = 0
    for row in train_matrix:
        nonzero_indicies = np.nonzero(row)
        per_20 = int(len(nonzero_indicies[0]) * 0.2)
        rand = random.choice(nonzero_indicies[0], per_20, replace=False)
        for i in range(per_20):
            test_matrix[u, rand[i]] = train_matrix[u, rand[i]]
            train_matrix[u, rand[i]] = 0
        u = u + 1
    return train_matrix, test_matrix


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def adj_cosine_sim(train_data):
    sims = np.zeros((N_MOVIE, N_MOVIE))
    # for i in range(N_MOVIE):
    #     for j in range(i, N_MOVIE):
    #         num = 0
    #         dem1 = 0
    #         dem2 = 0
    #         set_c_u = np.where((train_data[:, i] != 0) * (train_data[:, j]))[0]
    #         for k in set_c_u:
    #             num = num + sub_ratings[k][i] * sub_ratings[k][j]
    #             dem1 = dem1 + sub_ratings[k][i] ** 2
    #             dem2 = dem2 + sub_ratings[k][j] ** 2
    #             sim[i, j] = num / sqrt(dem1 * dem2 + 10 ** -12)

    user_mean = train_data.sum(axis=1)/(train_data != 0).sum(axis=1)

    sub_ratings = np.where(
        (train_data != 0), train_data - user_mean[:, None], train_data)
    for i in range(N_MOVIE):
        for j in range(i, N_MOVIE):
            sim = cosine_sim(sub_ratings[i], sub_ratings[j])
            sims[i, j] = sim
            sims[j, i] = sim
        # print i

    print(sims)

    return sims


def similarities(train_matrix):
    sim_matrix = adj_cosine_sim(train_matrix)

    upper_tri = np.triu(sim_matrix, k=1)
    upper_tri = upper_tri.T
    sim_matrix = sim_matrix + upper_tri

    sim_matrix = np.where((sim_matrix < 0), 0, sim_matrix)
    return sim_matrix


def predictions(train_matrix, sim_matrix):
    top = train_matrix.dot(sim_matrix)
    bottom = np.zeros((N_USER, N_MOVIE))
    for user in range(N_USER):
        nonzeros = np.nonzero(train_matrix[user])
        for movie in range(N_MOVIE):
            bottom[user, movie] = (sim_matrix[movie, nonzeros]).sum()

    pred_matrix = top / bottom
    np.nan_to_num(pred_matrix, copy=False)
    return pred_matrix


def err_rmse(test_matrix, pred_matrix):
    mse = mean_squared_error(
        test_matrix[test_matrix != 0], pred_matrix[test_matrix != 0])
    rmse = sqrt(mse)
    return rmse


if __name__ == "__main__":
    train, test = init_data()
    print(f'done init data')
    sim = similarities(train)
    print(f'done sim')
    pred = predictions(train, sim)
    print(f'done pred')
    err = err_rmse(test, pred)
    print(f'RMSE = ', err)
