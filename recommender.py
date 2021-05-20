#!/usr/bin/env python3
# long term project
# 2018008513 Son Young-in

import sys
import numpy as np
import pandas as pd

MAX_UID = 943
MAX_MID = 1682
UNKNOWN_RATING = -1

train_name = ""
test_name = ""
train_data = []
uid_list = []
mid_list = []


def get_cmd_args():
    global train_name, test_name
    if len(sys.argv) != 3:
        print("Too many or less arguments. Terminating...")
        return -1

    train_name = sys.argv[1]
    test_name = sys.argv[2]

    print("2018008513================")
    print(f"Training file name: {train_name}")
    print(f"Test file name: {test_name}")
    print("Let's start!")
    return 0


def init_data():
    scan_train_data()
    reshape_train_data()


def scan_train_data():
    global train_data, uid_list, mid_list
    train_data = np.empty((0, 4), dtype=int)
    with open(train_name) as file:
        for line in file.readlines():
            char_in_line = line.rstrip().split('\t')
            train_data = np.vstack((train_data, [int(char) for char in char_in_line]))


def reshape_train_data():
    global train_data

    new_data = np.full([MAX_MID + 1, MAX_MID + 1], -1)

    for tuple in train_data:
        new_data[uid(tuple), mid(tuple)] = rating(tuple)

    train_data = new_data
    return


def uid(t):
    return t[0]


def mid(t):
    return t[1]


def rating(t):
    return t[2]


def rating_and_time_from_tuple(t):
    return np.array([t[2], t[3]])


if __name__ == "__main__":
    if get_cmd_args() == -1:
        exit()
    init_data()
