#!/usr/bin/env python3
import numpy as np

TRAIN_DATA = None
SIMILARITY_MATRIX = None
RECO_MATRIX = None


def cosine_similarity(u1, u2):
    return np.dot(u1, u2) / (np.linalg.norm(u1) * (np.linalg.norm(u2)))


def pearson_coefficient(u1, u2):
    return np.corrcoef(u1, u2)


def init_train_data():
    # read train data and make a rating matrix D.
    return


def get_similarity_matrix():
    # get neighbors for each items using every user's ratings.
    return


def get_recommendation_matrix():
    # pick several items to recommend!
    return


def test():
    # do test using test data
    return


if __name__ == "__main__":
    print("start!")

    init_train_data()
    get_similarity_matrix()
    get_recommendation_matrix()
    test()
