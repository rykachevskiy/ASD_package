from lib.reader import read_alleles_to_num
import numpy as np
import os

from sklearn.model_selection import LeaveOneOut

import argparse

from lib.config import *
import lib.row_cheker as rch

import time


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

    print("Starting on original table...")
    table_2x3 = rch.check_initial_table(table, case_control.astype(bool))

    assert all(table_2x3.sum(1).sum(1) == 428)

    loo = LeaveOneOut()

    target_rows_list = []
    target_pvals_list = []

    chi2_hash = dict()

    for train_index, test_index in loo.split(table):
        beg = time.time()
        rows, pvals = rch.check_loo_table(table_2x3,
                                          case_control[test_index],
                                          table[test_index].flatten(),
                                          chi2_hash,
                                          cut=args.rows_number)
        target_rows_list.append(rows)
        target_pvals_list.append(pvals)
        end = time.time()
        print("Done with index {} in {} seconds".format(test_index, end - beg))

    target_rows = np.concatenate([x.reshape(1,-1) for x in target_rows_list], 0)
    target_pvals = np.concatenate([x.reshape(1,-1) for x in target_pvals_list], 0)

    print(target_rows.shape)

    np.save(args.output_table_path, target_rows)
    np.save(args.output_table_path+"_pvals", target_pvals)
