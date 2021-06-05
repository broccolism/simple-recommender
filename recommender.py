#!/usr/bin/env python3
# long term project
# 2018008513 Son Young-in

import sys
import numpy as np
from factorizer import Factorizer


MAX_UID = 943
MAX_MID = 1682
N_UID = 944
N_MID = 1683
UNKNOWN_RATING = -1

TRAIN_NAME = ""
TEST_NAME = ""
TRAIN_DATA = []
UID_LIST = []
MID_LIST = []


def get_cmd_args():
    global TRAIN_NAME, TEST_NAME
    if len(sys.argv) != 3:
        print("Too many or less arguments. Terminating...")
        return -1

    TRAIN_NAME = sys.argv[1]
    TEST_NAME = sys.argv[2]

    print("2018008513================")
    print(f"Training file name: {TRAIN_NAME}")
    print(f"Test file name: {TEST_NAME}")
    print("Let's start!")
    return 0


def init_data():
    scan_train_data()
    reshape_train_data()


def scan_train_data():
    global TRAIN_DATA, UID_LIST, MID_LIST
    TRAIN_DATA = np.empty((0, 4), dtype=int)
    with open(TRAIN_NAME) as file:
        for line in file.readlines():
            char_in_line = line.rstrip().split('\t')
            TRAIN_DATA = np.vstack((TRAIN_DATA, [int(char) for char in char_in_line]))


def reshape_train_data():
    global TRAIN_DATA

    new_data = np.full([N_UID, N_MID], -1)

    for tuple in TRAIN_DATA:
        new_data[get_uid(tuple), get_mid(tuple)] = get_rating(tuple)

    TRAIN_DATA = new_data
    return


def pre_use_preference():
    pre_use = TRAIN_DATA.copy()

    for uid in range(N_UID):
        for mid in range(N_MID):
            if pre_use[uid, mid] > 0:
                pre_use[uid, mid] = 1
            else:
                pre_use[uid, mid] = 0
    return pre_use


def SVD():

    return


def get_uid(t):
    return t[0]


def get_mid(t):
    return t[1]


def get_rating(t):
    return t[2]


def get_rating_and_time_from_tuple(t):
    return np.array([t[2], t[3]])


def put_zero(M, percentile):
    global TRAIN_DATA
    until = int(N_MID * percentile)
    for i in range(N_UID):
        count = 0
        tuples = [(j, M[i, j]) for j in range(N_MID)]
        tuples_sorted = sorted(tuples, key=lambda tup: tup[1], reverse=True)

        for j, _ in tuples_sorted:
            if TRAIN_DATA[i, j] == UNKNOWN_RATING:
                TRAIN_DATA[i, j] = 0
            count += 1
            if count == until:
                break

    print(f"put zero injected total {count} of zeros.")
    return M


if __name__ == "__main__":
    if get_cmd_args() == -1:
        exit()
    init_data()

    P = pre_use_preference()
    pre_factorizer = Factorizer(P, 10, 0.005, 0.05, 30)
    predicted_P = pre_factorizer.do_factorize()

    with np.printoptions(threshold=np.inf):
        predicted_P = put_zero(predicted_P, 0.5)
