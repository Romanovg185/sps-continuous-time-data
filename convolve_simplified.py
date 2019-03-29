import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

### Free parameters ###
kernel_steepness = 0.100

"""
Convolve with a Epanechnikov kernel
NOTE: Time runs until 300s to assure that the kernel can nicely converge to 0 at the right edge
:param m: Raw input spikes in a square matrix, padded with np.nans
:param kernel_steepness: Parameter of kernel width
:returns: A list of tuples of the [(peak_area[0], peak_amplitudes[0], peak_onset_time[0], peak_end_time[0]), ...]
"""
def convolve_with_kernel_two_sigma(m):
    # Make a continuous (dt=0.001) sum of kernel-convolved spike onset times
    u = np.arange(-1*kernel_steepness, kernel_steepness + 0.001, 0.001)
    kernel = 4*kernel_steepness/3*(1 - (u/kernel_steepness)**2)
    t_axis = np.arange(0, 1000, 0.001)
    t = np.zeros((m.shape[1], len(t_axis)))
    y = np.zeros_like(t)
    t += t_axis
    for i in range(m.shape[1]): # with i as cell index
        indices_of_spike = (m[:, i]/0.001).astype(int)
        indices_of_spike_nonan = indices_of_spike[indices_of_spike >= 0]
        y[i, indices_of_spike_nonan] = 1
        y[i, :] = np.convolve(y[i, :], kernel, mode='same')
    z = np.sum(y, axis=0)
    mu = np.mean(z[z != 0])
    sigma = np.std(z[z != 0])
    threshold = mu + 2*sigma

    # Indentify peaks as regions of z such that all values are 2sigma significantly above the mean, ending when a value lower than the mean is reached
    peaks = [[]]
    indices_peak_starts = []
    indices_peak_ends = []
    for index, i in enumerate(z):
        if i < threshold:
            if len(peaks[-1]) != 0:
                indices_peak_ends.append(index)
                peaks.append([])
        else:
            if len(peaks[-1]) == 0:
                indices_peak_starts.append(index)
            peaks[-1].append(i)
    if len(peaks[-1]) == 0:
        peaks = peaks[:-1]

    # Obtaining returns
    peak_areas = [0.001*sum(i) for i in peaks]
    times_peak_starts = [0.001*(i-1) for i in indices_peak_starts]
    times_peak_ends = [0.001*(i-1) for i in indices_peak_ends]
    peak_amplitudes = [np.max(i) for i in peaks]
    return list(zip(peak_areas, peak_amplitudes, times_peak_starts, times_peak_ends))

"""
Find indices of significant regions of firing pattern
:param m_sample: Matrix of sample
:param prob: Fraction of peaks to be an event
:param minimum_peak_area: Size that a peak has to have to be considered an event
###
:returns: List of tuples of (begin, end) forall significant peaks
"""
def get_indices_arbitrary_overlap(m_sample, prob=1):
    n_samples = 5 # Number of samples of ground truth
    areas, amplitudes, starts, ends = zip(*convolve_with_kernel_two_sigma(m_sample))
    indices = list(zip(starts, ends))
    print(indices)
    
    ## Stored if I want to filter even more
    #minimum_peak_area = sorted(areas)[int(len(areas)*prob)]
    #left_edges = list(filter(lambda x: x[0] > minimum_peak_area, list(zip(areas, starts))))
    #left_edges = [i[1] for i in left_edges]
    #right_edges = list(filter(lambda x: x[0] > minimum_peak_area, list(zip(areas, ends))))
    #right_edges = [i[1] for i in right_edges]
    #indices = list(zip(left_edges, right_edges))
    #indices = [(int(1000*i), int(1000*j)) for i, j in indices]
    return indices

"""
Obtain all neurons that possibly could be part of the pattern
:param m: Matrix containing onset times in continuous time
:returns: Matrix of concatenated synchronous patterns
"""
def locate_indices_neuron_per_pattern(m_sample):
    ind = get_indices_arbitrary_overlap(m_sample)
    patterns = []
    for start, end in ind:
        single_pattern = []
        for i, cell in enumerate(m_sample.T):
            right_of_start = cell > start
            left_of_end = cell < end
            if np.any(np.logical_and(left_of_end, right_of_start)):
                single_pattern.append(i)
        print(single_pattern)
        patterns.append(tuple(single_pattern))
    z = np.zeros((len(ind), m_sample.shape[1])) #z[event, cell_partaking]
    for i, cells in enumerate(patterns):
        z[i, cells] = 1
    return ind, z.T


def main():
    files = os.popen('ls Data').read().split('\n')[:-1]
    files = list({i[3:] for i in files})
    cortex_files = ['ctx' + i for i in files]
    cerebellum_files = ['cbl' + i for i in files]
    for file_name_cerebellum, file_name_cortex in zip(cerebellum_files, cortex_files):
        m_cbl = np.loadtxt('Data/' + file_name_cerebellum, delimiter=',')
        m_ctx = np.loadtxt('Data/' + file_name_cortex, delimiter=',')
        while m_cbl.shape[0] < m_ctx.shape[0]:
            to_stack = np.full((m_cbl.shape[1], 1), np.nan).T
            m_cbl = np.vstack([m_cbl, to_stack])
        while m_cbl.shape[0] > m_ctx.shape[0]:
            to_stack = np.full((m_ctx.shape[1], 1), np.nan).T
            m_ctx = np.vstack([m_ctx, to_stack])
        m_tot = np.hstack([m_cbl, m_ctx])
        participators, i = locate_indices_neuron_per_pattern(m_tot)
        np.savetxt(cortex_files + 'patterns.csv', participators, delimiter=',')
        np.savetxt(cortex_files + 'intervals_significant_correlation.csv', i, delimiter=',')

if __name__ == "__main__":
    main()
