import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import load_npz
from order_by_x_coord import load_mask_to_3d_matrix

def compute_jaccard_index(mask_0, mask_1, threshold=0.05):
    mask_0 = mask_0/np.max(mask_0) > threshold
    mask_1 = mask_1/np.max(mask_1) > threshold
    top = np.sum(np.logical_and(mask_0, mask_1))
    bot = np.sum(np.logical_or(mask_0, mask_1))
    jac = top/bot 
    return jac

def jaccard_matrix(filename):
    m_cbl = load_mask_to_3d_matrix(filename.replace('***', 'cbl'))
    m_ctx = load_mask_to_3d_matrix(filename.replace('***', 'ctx'))
    m = np.dstack([m_cbl, m_ctx])
    jacc_mat = np.zeros((m.shape[2], m.shape[2]))
    for i, mask_0 in enumerate(np.rollaxis(m, 2)):
        print(i)
        for j, mask_1 in enumerate(np.rollaxis(m, 2)):
            if i != j:
                jacc_mat[i, j] = compute_jaccard_index(mask_0, mask_1)
    return jacc_mat

def main():
    path = '/home/romano/Documents/ContinuousGlobalSynchrony/Masks/'
    result = '/home/romano/Documents/ContinuousGlobalSynchrony/JaccardIndexMatrices/'
    l = os.popen(f'ls {path}').read().split('\n')[:-1]
    l = {i[:6] + '***' + i[9:] for i in l}
    for i in l:
        j = jaccard_matrix(path + i)
        np.savetxt(result + i[:-3] + 'csv', j, delimiter=',')
    
        

if __name__ == '__main__':
    main()
