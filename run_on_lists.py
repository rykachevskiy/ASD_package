from lib.reader import read_alleles_to_num
import numpy as np
import os

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import argparse

from lib.config import *
from lib.utils import BySumClassifier, clean_table_2


if __name__ == '__main__':
    np.random.seed(239)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, dest='input_table_path')
    parser.add_argument('-r', type=str, dest='input_rows')

    parser.add_argument('-p', type=str, dest='predictions_path')
    args = parser.parse_args()

    table = np.load(args.input_table_path)
    rows_masks = np.load(args.input_rows)

    print(rows_masks.shape)
    case_control = np.array([0] * 407 + [1] * 21)

    loo = LeaveOneOut()

    predictions = np.zeros(HUMANS)

    for i, (train_index, test_index) in enumerate(loo.split(table)):
        curr_rows_mask = rows_masks[i]
        curr_table, _, _ = clean_table_2(table[:, curr_rows_mask])

        train_table = curr_table[train_index]
        test_table = curr_table[test_index]

        train_case_control = case_control[train_index]
        test_case_control = case_control[test_index]

        bsc = BySumClassifier()

        bsc.fit(train_table, train_case_control)
        predictions[i] = bsc.predict(test_table)

    for score,name in zip([f1_score, accuracy_score, roc_auc_score], ['f1', 'acc', 'auc']):
        print(name + ": ", score(case_control,predictions))


    np.save(args.predictions_path, predictions)

