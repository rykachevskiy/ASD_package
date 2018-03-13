from __future__ import division

import pandas as pd
import numpy as np
from lib.config import *


def clean_table_2(table):
    chains = []
    in_chains = set()
    if table.shape[1] > 0:
        t = np.zeros_like(table)

        for i in range(table.shape[1]):
            curr_chain = []
            if not i in in_chains:
                curr_chain.append(i)
                in_chains.add(i)

                for j in range(table.shape[1]):
                    if all(table[:, i] == table[:, j]) and not j in in_chains:
                        curr_chain.append(j)
                        in_chains.add(j)

                    # t = np.concatenate((t, table[:,i].reshape(-1,1)), axis = 1)
                t[:, i] = table[:, i]
            if len(curr_chain) != 0:
                chains.append(curr_chain)
        return t, chains, in_chains
    else:
        return table, chains, in_chains


class BySumClassifier:
    def __init__(self):
        pass
    def fit(self, X, y):
        self.sums = X.sum(1)
        self.mask = y.astype(bool)
        self.thrsh = np.mean([self.sums[np.logical_not(self.mask)].max(), self.sums[self.mask].min()])
    def predict(self, X):
        pred_sums = X.sum(1)
        return (pred_sums > self.thrsh).astype(int)