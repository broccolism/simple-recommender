#!/usr/bin/env python3
# long term project
# 2018008513 Son Young-in

import numpy as np

MAX_UID = 943
MAX_MID = 1682
N_UID = 944
N_MID = 1683


class Factorizer():
    def __init__(self, ratings, k, learning_rate, reg_param, iteration):
        self.R = ratings.copy()
        self.k = k
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.iteration = iteration
        self.init_more()

    def init_more(self):
        self.P = np.random.normal(scale=1./self.k, size=(N_UID, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(N_MID, self.k))

        self.B_user = np.zeros(N_UID)
        self.B_movie = np.zeros(N_MID)

        nonzero = self.R.nonzero()
        self.global_bias = np.mean(self.R[nonzero])

        rows, columns = nonzero
        self.nonzero_ratings = [(i, j, self.R[i, j]) for i, j in zip(rows, columns)]

    def do_factorize(self):
        self.train()
        return self.get_full_prediction()

    def train(self):
        for i in range(self.iteration):
            self.update()

    def get_error(self):
        rows, cols = self.R.nonzero()
        errors = []
        for x, y in zip(rows, cols):
            prediction = self.get_one_prediction(x, y)
            errors.append(self.R[x, y] - prediction)

        errors = np.array(errors)
        return np.sqrt(np.mean(errors ** 2))

    def get_one_prediction(self, i, j):
        prediction = np.dot(self.P[i, :], self.Q[j, :].T) + self.global_bias + self.B_user[i] + self.B_movie[j]
        return prediction

    def update(self):
        for i, j, r in self.nonzero_ratings:
            prediction = self.get_one_prediction(i, j)
            e = (r - prediction)

            self.B_user[i] += self.learning_rate * (e - self.reg_param * self.B_user[i])
            self.B_movie[j] += self.learning_rate * (e - self.reg_param * self.B_movie[j])

            self.P[i, :] += self.learning_rate * (e * self.Q[j, :] - self.reg_param * self.P[i, :])
            self.Q[j, :] += self.learning_rate * (e * self.P[i, :] - self.reg_param * self.Q[j, :])

    def get_full_prediction(self):
        return np.dot(self.P, self.Q.T) + self.global_bias + self.B_user[:, np.newaxis] + self.B_movie[np.newaxis, :]
