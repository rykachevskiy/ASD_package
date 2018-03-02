from lib.reader import read_alleles_to_num
import numpy as np
import os

from sklearn.model_selection import LeaveOneOut

import argparse

from lib.config import *
from lib.row_cheker import check_table


if __name__ == '__main__':
    np.random.seed(239)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, dest='input_table_path')
    parser.add_argument('-o', type=str, dest='output_table_path')
    parser.add_argument('-n', type=int, dest='rows_number', default=500)
    args = parser.parse_args()

    table = np.load(args.input_table_path)
    print(table.shape)
    case_control = np.array([0] * 407 + [1] * 21)

    loo = LeaveOneOut()

    target_rows_list = []

    for train_index, test_index in loo.split(table):
        train_table = table[train_index]
        test_table = table[test_index]

        train_case_control = case_control[train_index]
        test_case_control = case_control[test_index]

        target_rows_list.append(check_table(train_table, train_case_control, args.rows_number))

    target_rows = np.concatenate([x.reshape(-1,1) for x in target_rows_list], 1)

    print(target_rows.shape)

    np.save(args.output_table_path, target_rows)

