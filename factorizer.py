#!/usr/bin/env python3
# long term project
# 2018008513 Son Young-in

import numpy as np

MAX_UID = 943
MAX_MID = 1682
N_UID = 944
N_MID = 1683
UNKNOWN_RATING = -1

# Mask for uninteresting items.


class Factorizer:
    def __init__(self, P, k, r_lambda):
        self.P = P
        self.k = k
        self.r_lambda = r_lambda
        self.X = np.random.normal(size=(N_UID, self.k))
        self.Y = np.random.normal(size=(N_MID, self.k))
        self.init_W()

    def init_W(self):
        self.W = np.zeros(shape=(N_UID, N_MID))
        for i in range(N_UID):
            for j in range(N_MID):
                nonzero_ratio = len(np.nonzero(self.P[i])) / N_MID
                if self.P[i, j] > 0:
                    self.W[i, j] = 1
                else:
                    self.W[i, j] = nonzero_ratio

    def train(self):
        X = self.update_X(self.X, self.Y)
        print(X)
        return

    def update_X(self, X, Y):
        for i in range(N_UID):
            W_ = np.diag(self.W[i])
            RiWiV = np.matmul(np.matmul(self.P[i], W_), Y)
            VtWiV = np.matmul(np.matmul(np.transpose(Y), W_), Y)
            lambdaWI = self.r_lambda * np.sum(self.W[i]) * np.identity(self.k)
            inversed = np.linalg.inv((VtWiV + lambdaWI))

            X[i] = np.linalg.solve(RiWiV, inversed)
            print(X[i])
        print(X)
        return X

    def update_Y(self, X, Y):
        return

    def get_full_prediciton(self):
        return
