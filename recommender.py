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
UNKNOWN_RATING = 0

TRAIN_NAME = ""
TEST_NAME = ""
RATINGS = []
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
    global RATINGS, UID_LIST, MID_LIST
    RATINGS = np.empty((0, 4), dtype=int)
    with open(TRAIN_NAME) as file:
        for line in file.readlines():
            char_in_line = line.rstrip().split('\t')
            RATINGS = np.vstack((RATINGS, [int(char) for char in char_in_line]))


def reshape_train_data():
    global RATINGS

    new_data = np.full([N_UID, N_MID], UNKNOWN_RATING)

    for tuple in RATINGS:
        new_data[get_uid(tuple), get_mid(tuple)] = get_rating(tuple)

    RATINGS = new_data
    return


def get_uid(t):
    return t[0]


def get_mid(t):
    return t[1]


def get_rating(t):
    return t[2]


def print_output(predicted):
    file_name = f"{TRAIN_NAME.split('.')[0]}.base_prediction.txt"
    with open(file_name, 'w') as output, open(TEST_NAME) as test:
        for line in test.readlines():
            char_in_line = line.rstrip().split('\t')
            uid = int(get_uid(char_in_line))
            mid = int(get_mid(char_in_line))
            rating = predicted[uid, mid]
            print_one_line(output, uid, mid, rating if rating > 0 else 1.)


def print_one_line(file, uid, mid, rating):
    line = f"{uid}\t{mid}\t{rating}\n"
    file.write(line)


if __name__ == "__main__":
    if get_cmd_args() == -1:
        exit()
    init_data()

    factorizer = Factorizer(RATINGS, 10, 0.005, 0.05, 100)
    # 0.005, 0.05 - 1000: 0.96 (34m) # 100: 0.924776 (2m), 0.9344228 (2m)
    # 0.002, 0.05 - 100: 0.9533, 0.9491, 0.9508899
    predicted = factorizer.do_factorize()
    print_output(predicted)
    print("DONE!")
