#!/usr/bin/env python3
# long term project
# 2018008513 Son Young-in

import numpy as np
from numpy.core.fromnumeric import mean

MAX_UID = 943
MAX_MID = 1682
N_UID = 944
N_MID = 1683
UNKNOWN_RATING = -1

# Mask for uninteresting items.


class Factorizer:
    def __init__(self, P, k, alpha, beta, threshold):
        self.P = P
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.X = np.random.normal(size=(N_UID, self.k))
        self.Y = np.random.normal(size=(N_MID, self.k))
        self.init_bias()

    def init_bias(self):
        self.B_user = np.zeros(N_UID)
        self.B_movie = np.zeros(N_MID)
        self.b_global = np.mean(self.P[np.nonzero(self.P)])

    def mean_err(self):
        rows, cols = np.nonzero(self.P)
        predicted = self.get_full_prediction()
        err_sum = 0
        for row, col in zip(rows, cols):
            err_sum += np.power((self.P[row, col] - predicted[row, col]), 2)
        return np.sqrt(err_sum) / len(rows)

    def train(self):
        for i in range(10):
            self.update()
            cur_err = self.mean_err()
            print(round(cur_err, 10))
        return self.get_full_prediction()

    def update(self):
        for i in range(N_UID):
            for j in range(N_MID):
                err = self.P[i, j] - self.get_predicted_rating(i, j)

                self.B_user[i] += self.alpha * (err - self.beta * self.B_user[i])
                self.B_movie[j] += self.alpha * (err - self.beta * self.B_movie[j])

                X_i = self.X[i, :][:]
                self.X[i, :] += self.alpha * (err * self.Y[j, :] - self.beta * self.X[i, :])
                self.Y[j, :] += self.alpha * (err * X_i - self.beta * self.Y[j, :])

    def get_predicted_rating(self, i, j):
        return np.dot(self.X[i, :], self.Y[j, :].T) + self.b_global + self.B_user[i] + self.B_movie[j]

    def get_full_prediction(self):
        return np.dot(self.X, self.Y.T) + self.b_global + self.B_user[:, np.newaxis] + self.B_movie[np.newaxis, :]
