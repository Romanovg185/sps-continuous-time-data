import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

def main():
    n = 10
    files = os.popen('ls FourBoxRawData').read().split('\n')[:-1]
    for my_file in files:
        results = []
        data = np.loadtxt('FourBoxRawData/' + my_file, delimiter=',')
        n_cbl = np.where(np.isnan(data[0, :]))[0][0]
        cbl_v_cbl = np.tril(data[:n_cbl, :n_cbl])
        ctx_v_ctx = np.tril(data[n_cbl+1:, n_cbl+1:])
        cbl_v_ctx = np.tril(data[:n_cbl, n_cbl+1:])
        for i in range(n):
            m0 = np.max(cbl_v_cbl)
            m1 = np.max(ctx_v_ctx)
            m2 = np.max(cbl_v_ctx)
            if m0 >= m1 and m0 >= m2:
                indices = np.argwhere(cbl_v_cbl == m0)
                if len(indices) > 1:
                    indices = [indices[0]]
                results.append(('cbl{}'.format(indices[0][0]), 'cbl{}'.format(indices[0][1]), m0)) 
                cbl_v_cbl[indices] = 0
            elif m1 >= m0 and m1 >= m2:
                indices = np.argwhere(ctx_v_ctx == m1)
                if len(indices) > 1:
                    indices = [indices[0]]
                results.append(('ctx{}'.format(indices[0][0]), 'ctx{}'.format(indices[0][1]), m0)) 
                ctx_v_ctx[indices] = 0
            elif m2 >= m0 and m2 >= m1:
                indices = np.argwhere(cbl_v_ctx == m2)
                if len(indices) > 1:
                    indices = [indices[0]]
                results.append(('cbl{}'.format(indices[0][0]), 'ctx{}'.format(indices[0][1]), m0)) 
                cbl_v_ctx[indices] = 0
        print(results)


if __name__ == '__main__':
    main()
