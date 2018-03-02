import numpy as np
from scipy.stats import chi2_contingency


def to_table_2x3(table):
    return table


def to_0vs12(table):
    new_table = np.zeros((2, 2))

    new_table[:, 0] += table[:, 0]
    new_table[:, 1] += table[:, 1] + table[:, 2]

    return new_table


def to_01vs2(table):
    new_table = np.zeros((2, 2))

    new_table[:, 0] += table[:, 0] + table[:, 1]
    new_table[:, 1] += table[:, 2]

    return new_table


def to_02vs1(table):
    new_table = np.zeros((2, 2))

    new_table[:, 1] = table[:, 1]
    new_table[:, 0] += table[:, 0] + table[:, 2]

    return new_table


def to_allelic(table):
    new_table = np.zeros((2, 2))

    new_table[:, 1] = table[:, 1] + 2 * table[:, 2]
    new_table[:, 0] += table[:, 1] + 2 * table[:, 0]

    return new_table

def table_wrap(table):
    if (table.sum(0) == 0).sum() > 0:
        if table.shape[1] == 3:
            table = to_0vs12(table)
    if (table.sum(0) == 0).sum() > 0:
        table = np.ones((2, 2))

    return table


def check_p_vals(column, case_control_mask, tests_mask=[True]*5):
    table = np.zeros((2, 3))

    table[0][0] = (column[case_control_mask] == 0).sum()
    table[0][1] = (column[case_control_mask] == 1).sum()
    table[0][2] = (column[case_control_mask] == 2).sum()

    table[1][0] = (column[np.logical_not(case_control_mask)] == 0).sum()
    table[1][1] = (column[np.logical_not(case_control_mask)] == 1).sum()
    table[1][2] = (column[np.logical_not(case_control_mask)] == 2).sum()
    p = np.ones(5)

    for i, table_creator in enumerate([to_table_2x3, to_0vs12, to_01vs2, to_02vs1, to_allelic]):
        if tests_mask[i]:
            new_table = table_wrap(table_creator(np.copy(table)))
            p[i] = chi2_contingency(new_table, True)[1]


    return p

def check_table(table, case_contol_mask, cut=500):
    min_p_vals = np.ones(table.shape[1])

    for i in range(table.shape[1]):
        p_vals = check_p_vals(table[:, i], case_contol_mask, [False, False, False, False, True])
        min_p_vals[i] = p_vals.min()

    return min_p_vals.argsort()[:cut], np.sort(min_p_vals)[:cut]


