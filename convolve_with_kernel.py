import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

### Free parameters ###
width_sampling_frame_ground_truth_permutation = 1
kernel_steepness = 0.100
cutoff_prob_spurious_selection = 0.05

"""
Produces a ground truth by taking width-sized intervals of random cells, sticking them together to get a ground truth. Width should be representative for at least 2x the average plateau width
:param m: Matrix with raw data to infer from
:param width: Width of time intervals sampled
:returns: A ground truth matrix
"""
def make_permutation_based_ground_truth(m):
    width = width_sampling_frame_ground_truth_permutation
    m = m.T
    total = []
    max_t = np.nanmax(m)
    for virtual_cell in range(m.shape[0]):
        per_cell = set() # Set (inherently singleton) used to prevent double sampling of points
        for i, random_cell in enumerate(np.random.randint(0, m.shape[0], np.ceil(max_t/width).astype(int))):
            trace = m[random_cell, :][np.logical_not(np.isnan(m[random_cell, :]))]
            random_interval_start = max_t*np.random.rand()
            start = random_interval_start
            end = random_interval_start + width
            left = trace >= start
            right = trace < end
            mask = np.logical_and(left, right)
            segment = trace[mask]
            segment -= random_interval_start
            segment += i*width
            per_cell.update((i for i in segment)) # Passing in a generator because I'm cool like that
        per_cell = sorted(list(per_cell))
        total.append(per_cell)

    # To rectangular matrix
    max_size = max([len(i) for i in total])
    m_ground = np.full((m.shape[0], max_size), np.nan)
    for i, el in enumerate(total):
        m_ground[i, :len(el)] = np.array(el)

    return m_ground.T

"""
Convolve with a Epanechnikov kernel
NOTE: Time runs until 300s to assure that the kernel can nicely converge to 0 at the right edge
:param m: Raw input spikes in a square matrix, padded with np.nans
:param kernel_steepness: Parameter of kernel width
:returns: A list of tuples of the [(peak_area[0], peak_amplitudes[0], peak_onset_time[0], peak_end_time[0]), ...]
"""
def convolve_with_kernel(m):
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
    z -= np.mean(z)
    z[z < 0] = 0

    # Indentify peaks as regions of z such that all values are above the mean, ending when a value lower than the mean is reached
    peaks = [[]]
    indices_peak_starts = []
    indices_peak_ends = []
    for index, i in enumerate(z):
        if i == 0:
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
Determine minimum area width to be considered an SP as the area that only appears randomly under H0 with a p=0.05
:param m_ground: Ground truth matrix
:returns: Float that contains the minimum peak area for it to be considered a spike
"""
def find_minimum_peak_area(m_ground):
    l = convolve_with_kernel(m_ground)
    areas, amplitudes, onset_times, end_times = zip(*l) # From list of 4-tuples to 4 tuples
    sorted_areas = np.sort(areas)
    min_area = sorted_areas[np.floor((1 - cutoff_prob_spurious_selection)*len(sorted_areas)).astype(int)]
    return min_area

"""
Find indices of significant regions of firing pattern
:param m_sample: Matrix of sample
:param minimum_peak_area: Size that a peak has to have to be considered an event
###
:returns: List of tuples of (begin, end) forall significant peaks
"""
def get_indices_significant_overlap(m_sample):
    n_samples = 5 # Number of samples of ground truth
    l = list(map(find_minimum_peak_area, [make_permutation_based_ground_truth(m_sample) for i in range(n_samples)]))
    minimum_peak_area = sum(l)/len(l)
    areas, amplitudes, starts, ends = zip(*convolve_with_kernel(m_sample))
    left_edges = list(filter(lambda x: x[0] > minimum_peak_area, list(zip(areas, starts))))
    left_edges = [i[1] for i in left_edges]
    right_edges = list(filter(lambda x: x[0] > minimum_peak_area, list(zip(areas, ends))))
    right_edges = [i[1] for i in right_edges]
    indices = list(zip(left_edges, right_edges))
    indices = [(int(1000*i), int(1000*j)) for i, j in indices]
    return indices

"""
Obtain all neurons that possibly could be part of the pattern
:param m: Matrix containing onset times in continuous time
:returns: Matrix of concatenated synchronous patterns
"""
def locate_indices_neuron_per_pattern(m_sample):
    ind = get_indices_significant_overlap(m_sample)
    patterns = []
    for start, end in ind:
        single_pattern = []
        for i, cell in enumerate(m_sample.T):
            right_of_start = cell > start/1000
            left_of_end = cell < end/1000
            if np.any(np.logical_and(left_of_end, right_of_start)):
                single_pattern.append(i)
        patterns.append(tuple(single_pattern))
    z = np.zeros((len(ind), m_sample.shape[1])) #z[event, cell_partaking]
    for i, cells in enumerate(patterns):
        z[i, cells] = 1
    return z.T

def identify_patterns_two_samples(m_0, m_1):
    def reshape(m, shape):
        m_reshaped = np.full((shape[0], m.shape[1]), np.nan)
        m_reshaped[:m.shape[0], :] = m
        return m_reshaped

    def get_cells_firing_between_ts(m, left, right):
        cells = []
        for i, neuron in enumerate(m):
            if len(np.where(np.logical_and(neuron >= left, neuron < right))[0]) > 0:
                cells.append(i)
        final = np.zeros((m.shape[0], 1))
        final[cells] = 1
        return final

    n_samples = 1
    ground_0 = [make_permutation_based_ground_truth(m_0) for i in range(n_samples)]
    ground_1 = [make_permutation_based_ground_truth(m_1) for i in range(n_samples)]
    ground_0_r = [reshape(i, j.shape) if i.shape[0] < j.shape[0] else i for i, j in zip(ground_0, ground_1)]
    ground_1_r = [reshape(j, i.shape) if j.shape[0] < i.shape[0] else j for i, j in zip(ground_0, ground_1)]
    peak_data = [convolve_with_kernel(np.concatenate([i, j], axis=1).T) for i, j in zip(ground_0_r, ground_1_r)]
    
    # i[0] corresponds to peak area, here we flatten based on that
    flat_peak_area = []
    for sample in peak_data:
        for point in sample:
            flat_peak_area.append(point[0])
    sorted_peak_area = list(sorted(flat_peak_area))
    area_generator = (el for i, el in enumerate(sorted_peak_area) if (1 - cutoff_prob_spurious_selection) < i/len(sorted_peak_area))
    min_area = next(area_generator) # Generates first element in area_generator, i.e. the first element that's large enough
    
    m_0_r = reshape(m_0, m_1.shape) if m_0.shape[0] < m_1.shape[0] else m_0
    m_1_r = reshape(m_1, m_0.shape) if m_1.shape[0] < m_0.shape[0] else m_1
    peak_data = convolve_with_kernel(np.concatenate([m_0_r, m_1_r], axis=1).T)
    peak_data_sorted = sorted(peak_data, key=lambda x: x[0])
    time_intervals_of_significance = [(i[2], i[3], i[0]) for i in peak_data_sorted if i[0] > min_area]
    print(time_intervals_of_significance)

    involved_vectors = [get_cells_firing_between_ts(np.concatenate([m_0_r, m_1_r], axis=1).T, i[0], i[1]) for i in time_intervals_of_significance]
    involved_areas = [i[2] for i in time_intervals_of_significance]
    involved_vector_areas = [i for i in zip(involved_areas, involved_vectors)]
    involved_vector_areas_sorted = sorted(involved_vector_areas, key=lambda x: x[0], reverse=True)
    involved_areas = [i[0] for i in involved_vector_areas_sorted]
    involved_vectors = [i[1] for i in involved_vector_areas_sorted]
    
    involved_matrix = np.hstack(involved_vectors)
    involved_matrix = np.vstack([np.array(involved_areas).T, involved_matrix])

    area_strengths = involved_matrix[0, :]
    first_signal = involved_matrix[1:m_0.shape[1]+1, :]
    second_signal = involved_matrix[m_0.shape[1]+1:, :]
    
    return area_strengths, first_signal, second_signal

    
def plot_spectrum():
    file_name_cerebellum = 'Scope1_denoised_mc_results.csv'
    file_name_cortex = 'Scope2_denoised_mc_results.csv'
    weight_factor_for_detecting_SPs = 1

    for i, shift in enumerate(0.25*np.arange(-4, 5)):
        print(shift)
        m_cbl = np.loadtxt(file_name_cerebellum, delimiter=',')
        m_ctx = np.loadtxt(file_name_cortex, delimiter=',')
        m_ctx += shift
        p_cbl, p_ctx = identify_patterns_two_samples(m_cbl, m_ctx)
        spectrum = p_cbl[0, :]
        plt.semilogy(spectrum, label=r"$\phi_{\mathrm{cortex}} = $" + str(round(shift, 2)) + " sec", c='C{}'.format(i))
    plt.xlabel('Sorted spectrum index')
    plt.ylabel('Area of detected convolution peak')
    plt.legend()
    plt.show()

def plot_areas_involved():
    file_name_cerebellum = 'Scope1_denoised_mc_results.csv'
    file_name_cortex = 'Scope2_denoised_mc_results.csv'
    shifts = np.arange(-4, 4.01, 0.01)
    weight_factor_for_detecting_SPs = 1

    m_cbl = np.loadtxt(file_name_cerebellum, delimiter=',')
    m_ctx = np.loadtxt(file_name_cortex, delimiter=',')
    l = [(m_cbl, m_ctx + shift) for shift in shifts]
    def helper(x): return identify_patterns_two_samples(x[0], x[1])
    with Pool(3) as p:
        ll = p.map(helper, l)
    spectra = [np.sum(i[0, :]) + np.sum(j[0, :]) for i, j in ll]
    with open('output.txt', 'w') as f:
        for i in spectra:
            f.write(str(i))
            f.write(', ')


def helper(x):
    return identify_patterns_two_samples(x[0], x[1])

def plot_n_neurons_involved():
    files = os.popen('ls Data').read().split('\n')[:-1]
    files = list({i[3:] for i in files})
    cortex_files = ['ctx' + i for i in files]
    cerebellum_files = ['cbl' + i for i in files]
    #for file_name_cortex, file_name_cerebellum in zip(cortex_files, cerebellum_files)
    #m_cbl = np.loadtxt(file_name_cerebellum, delimiter=',')
    #m_ctx = np.loadtxt(file_name_cortex, delimiter=',')
    #ll = [helper(m_cbl, m_ctx))
    #spectra = [np.sum(np.sum(i[1:, :], axis=0)) + np.sum(np.sum(i[1:, :], axis=0)) for i, j in ll]
    #with open('output.txt', 'w') as f:
    #    for i in spectra:
    #        f.write(str(i))
    #        f.write(', ')

if __name__ == "__main__":
    files = os.popen('ls Data').read().split('\n')[:-1]
    files = list({i[3:] for i in files})
    cortex_files = ['ctx' + i for i in files]
    cerebellum_files = ['cbl' + i for i in files]
    file_name_cortex = 'cbl_16082018_153856.csv'
    file_name_cerebellum = 'ctx_16082018_153856.csv'
    m_cbl = np.loadtxt('Data/' + file_name_cerebellum, delimiter=',')
    sorted_areas = []
    for i in range(20):
        print(i)
        m_ground = make_permutation_based_ground_truth(m_cbl)
        l = convolve_with_kernel(m_ground)
        areas, amplitudes, onset_times, end_times = zip(*l) # From list of 4-tuples to 4 tuples
        sorted_areas.extend(areas)
    plt.hist(sorted_areas, label='Ground truth', facecolor=(1, 0, 0, 0.2), bins=50)
    l = convolve_with_kernel(m_cbl)
    areas, amplitudes, onset_times, end_times = zip(*l) # From list of 4-tuples to 4 tuples
    sorted_areas = np.sort(areas)
    plt.hist(sorted_areas, label='Real data', facecolor=(0, 0, 1, 0.2), bins=50)
    plt.legend()
    plt.show()
    

