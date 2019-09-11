import os
import numpy as np
import matplotlib.pyplot as plt

### Free parameters ###
kernel_steepness = 0.100

def convolve_cell(cell):
    u = np.arange(-1*kernel_steepness, kernel_steepness + 0.01, 0.01)
    kernel = 4*kernel_steepness/3*(1 - (u/kernel_steepness)**2)
    t = np.arange(0, 300, 0.01)
    y = np.zeros_like(t)
    indices_of_spike = (cell/0.01).astype(int)
    indices_of_spike_nonan = indices_of_spike[indices_of_spike >= 0]
    y[indices_of_spike_nonan] = 1
    y = np.convolve(y, kernel, mode='same')
    return y

def get_kernel_sum(m):
    kernel_sum = []
    for i in m.T:
        kernel_sum.append(convolve_cell(i))
    return np.sum(np.vstack(kernel_sum), axis=0)

"""
Permutation-based surrogate construction
"""
def make_surrogate(m_raw):
    surrogates = []
    for h, i in enumerate(np.random.randint(0, m_raw.shape[1], m_raw.shape[1])):
        cell = m_raw[:, i]
        cell = cell[~np.isnan(cell)]
        total_max = np.nanmax(np.nanmax(m_raw))
        n_bins = int(total_max/0.01)
        ts = np.zeros(n_bins+1)
        indices = tuple((cell * 1/0.01).astype(int))
        for j in indices:
            ts[j] = 1
        np.random.shuffle(ts)
        indices = np.where(ts == 1)[0]
        surrogates.append(indices/100)
    max_len = max([len(i) for i in surrogates])
    ret = np.full((max_len, m_raw.shape[1]), np.nan)
    for i, el in enumerate(surrogates):
        ret[:len(el), i] = el
    return ret

"""
Return the DF/F criterion for synchronous event selection
:param m_raw: Matrix of inferred onset times to base the ground truth on
"""
def get_ground_truth_threshold_criterion(m_cbl, m_ctx, n_standard_deviations, n=3):
    zs = []
    for i in range(n):
        sur_cbl = make_surrogate(m_cbl)
        sur_ctx = make_surrogate(m_ctx)
        z_cbl = get_kernel_sum(sur_cbl)
        z_ctx = get_kernel_sum(sur_ctx)
        z = z_cbl + z_ctx
        zs.append(z)
    trimmed_data = [np.trim_zeros(i, 'b') for i in zs]
    mu = np.mean([np.mean(i) for i in trimmed_data])
    sigma = np.mean([np.std(i) for i in trimmed_data])
    return mu + n_standard_deviations*sigma

def try_varying_ground_truths(start, end, step):
    for my_file in os.listdir('FRIOnsetTimes'):
        if 'cbl' in my_file:
            continue
        m_ctx = np.loadtxt('FRIOnsetTimes/' + my_file, delimiter=',')
        m_cbl = np.loadtxt('FRIOnsetTimes/cbl' + my_file[3:], delimiter=',')
        l = []
        for sigma in np.arange(start, end, step):
            l.append(get_ground_truth_threshold_criterion(m_cbl, m_ctx, sigma))
        m = np.vstack([np.arange(start, end, step), np.array(l)])
        np.savetxt('Thresholds/' + my_file[12:], m, delimiter=',')
    

if __name__ == "__main__":
    m_cbl = np.loadtxt('/home/romano/Desktop/sps-continuous-time-data-master/FRIOnsetTimes/cbl_results_16082018_132308.csv', delimiter=',')
    m_ctx = np.loadtxt('/home/romano/Desktop/sps-continuous-time-data-master/FRIOnsetTimes/ctx_results_16082018_132308.csv', delimiter=',')
    for i in range(1, 10):
        print(get_ground_truth_threshold_criterion(m_cbl, m_ctx, i))
