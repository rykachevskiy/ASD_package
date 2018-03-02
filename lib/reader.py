from __future__ import division

import pandas as pd
import numpy as np
from lib.config import *

def read_alleles_to_num(low, high, mask, f, allele_list=None, use_pd=True, print_interval=1000000):
    n = sum(mask[:high])

    header = f.read(3)

    if use_pd:
        t = pd.DataFrame(columns=np.arange(n), index=np.arange(HUMANS), dtype=np.int8)
    else:
        t = np.zeros((HUMANS, n)).astype(np.int8)

    pos_in_t = 0

    f.seek(HUMANS // HUMANS_IN_BYTE * low, 1)

    for allele_num in range(low, high):
        SNP = f.read(HUMANS // HUMANS_IN_BYTE)

        if allele_num % print_interval == 0 and allele_num != 0:
            print('allele num: ', allele_num) #no tqdm now

        if mask[allele_num - low]:
            for k in range(HUMANS // HUMANS_IN_BYTE):
                b = SNP[k]
                for i in range(HUMANS_IN_BYTE):
                    curr = b & 3

                    human, pos = k * HUMANS_IN_BYTE + i, pos_in_t

                    if use_pd:
                        x, y = pos, human
                    else:
                        x, y = human, pos


                    if curr == 3:
                        t[x][y] = 0
                    elif curr == 2:
                        t[x][y] = 1
                    elif curr == 0:
                        t[x][y] = 2
                    else:
                        t[x][y] = -1

                    b = b >> 2

            if not allele_list is None:
                t.rename(index=str, columns={pos_in_t: allele_list[allele_num]}, inplace=True)

            pos_in_t += 1
    return t