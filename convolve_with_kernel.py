import numpy as np
import matplotlib.pyplot as plt

####################################
# GROUND TRUTHS NOT YET FUNCTIONAL #
####################################
"""
Makes a ground hypothesis, assuming all cells fire uncorrelated and with a rate equal to m
:param m: Matrix to infer firing rate fram
:param width: Not used, just to obtain same input type as make_cortical_ground_hypothesis
:returns: A ground truth firing matrix
"""
def make_ground_hypothesis(m, width):
    # Infer inter-spike time interval iti
    itis = []
    for i in m.T:
        spikes = i[np.logical_not(np.isnan(i))]
        iti = spikes[1:] - spikes[:-1]
        itis.extend(iti)
    gram = sorted(itis) - min(itis)
    lambda_hat = 1/np.mean(gram) # Maximum likelihood estimator of exponential distribution is 1/lambda as per Wiki

    # Generate ground hypothesis based on lambda_hat
    firing_times = []
    for i in m.T:
        max_t = np.nanmax(i)
        in_silico = np.random.exponential(scale=lambda_hat, size=(10000, 1))
        firing_time = np.cumsum(in_silico)
        firing_time = firing_time[firing_time < max_t]
        firing_times.append(firing_time)
    max_num_firing_times = max([len(i[np.logical_not(np.isnan(i))]) for i in firing_times])
    firing_matrix = np.full((len(firing_times), max_num_firing_times), np.nan)
    for i, el in enumerate(firing_times):
        firing_matrix[i, :len(el)] = el
    return(firing_matrix.T)

"""
Generate a ground hypothesis, assuming cortical data consisting of a summation of sparse spikes and firing plateaus
:param m: Matrix to infer from
param width: Time width of a group to consider it a plateau
:returns: A ground truth firing matrix
"""
def make_cortical_ground_hypothesis(m, width):
    min_number_for_plateau = 2
    lam_sparse = 0
    lam_plateau = 0
    n_plateau = 0
    singleton_distances = []
    plateau_distances = []
    plateau_n = []
    plateau_spikes = []
    for cell in m.T:
        cell = cell[np.logical_not(np.isnan(cell))]
        groupings = [[0.0]]
        for spike in cell:
            if abs(spike - groupings[-1][-1]) < width:
                groupings[-1].append(spike)
            else:
                groupings.append([spike])
        singletons = [i for i in groupings if len(i) == min_number_for_plateau]
        singleton_distance = [abs(i[0] - j[0]) for i, j in zip(singletons[1:], singletons[:-1])]
        plateaus = [i for i in groupings if len(i) > min_number_for_plateau]
        plateau_distance = list(map(lambda x: abs(x[0][0] - x[1][0]), zip(plateaus[1:], plateaus[:-1])))
        plateau_number = [len(i) for i in plateaus]
        l = []
        for plat in plateaus:
            l.extend([abs(i - j) for i, j in zip(plat[1:], plat[:-1])])
        singleton_distances.extend(singleton_distance)
        plateau_distances.extend(plateau_distance)
        plateau_n.extend(plateau_number)
        plateau_spikes.extend(l)

    d_singleton_hat = np.mean(singleton_distances)
    d_plateau_hat = np.median(plateau_distances)
    n_peaks_in_plateau_hat = np.median(plateau_distance)
    d_plateau_spikes_hat = np.mean(plateau_spikes)

    # Sparse spikes
    m_singleton = []
    for i in range(m.shape[1]):
        singleton_spikes = np.cumsum(np.random.exponential(d_singleton_hat, size=10000))
        z = singleton_spikes < np.nanmax(m[:, i])
        singleton_spikes = singleton_spikes[z] # Not larger than original time series
        m_singleton.append(singleton_spikes)

    # Plateaus
    m_plat = []
    #for i in range(m.shape[1]):
    #    plateau_start_times = np.cumsum(np.random.exponential(d_plateau_hat, size=1000))
    #    plateau_start_times = plateau_start_times[plateau_start_times < np.nanmax(m[i,:])]
    #    plateau_spikes = [np.cumsum(np.random.exponential(d_plateau_spikes_hat, size=i.astype(int))) for i in np.random.exponential(d_plateau_hat, size=plateau_start_times.size)]
    #    plateau_final = [i + j for i, j in zip(plateau_start_times, plateau_spikes)]
    #    plateau_final = np.hstack(plateau_final)
    #    m_plat.append(plateau_final)

    # Combining
    #m_tot = [sorted(np.hstack([i, j])) for i, j in zip(m_singleton, m_plat)]
    m_tot = [sorted(i) for i in m_singleton]
    l_tot = max([len(i) for i in m_tot])
    final = np.full((m.shape[1], l_tot), np.nan)
    for i, el in enumerate(m_tot):
        final[i, :len(el)] = el
    for i, el in enumerate(final.T):
        plt.scatter(el, i*np.ones_like(el), c='C0')
    plt.show()
    return final.T


###################################
# CODE FUNCTIONAL FROM THIS POINT #
###################################

"""
Convolve with a Epanechnikov kernel
:param m: Raw input spikes in a square matrix, padded with np.nans
:param kernel_steepness: Parameter of kernel width
:returns: A tuple containing per peak ([peak_area[0], ...], [peak_amplitudes], [peak_onset_time], [peak_end_time])
"""
def convolve_with_kernel(m, kernel_steepness=0.100):
    # Make a continuous (dt=0.001) sum of kernel-convolved spike onset times
    u = np.arange(-1*kernel_steepness, kernel_steepness + 0.001, 0.001)
    kernel = 4*kernel_steepness/3*(1 - (u/kernel_steepness)**2)
    t_axis = np.arange(0, np.ceil(np.nanmax(m)), 0.001)
    t = np.zeros((m.shape[0], len(t_axis)))
    y = np.zeros_like(t)
    t += t_axis
    for i in range(m.shape[1]):
        indices_of_spike = (m[:, i]/0.001).astype(int)
        indices_of_spike_nonan = indices_of_spike[indices_of_spike >= 0]
        y[i, indices_of_spike_nonan] = 1
        y[i, :] = np.convolve(y[i, :], kernel, mode='same')
    z = np.sum(y, axis=0)

    # Indentify peaks as regions of z such that all values are above zero, ending when a zero is reached
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
    return peak_areas, peak_amplitudes, times_peak_starts, times_peak_ends

"""
Determine minimum area width to be considered an SP as the area that only appears randomly under H0 with a p=0.05
:param m_ground: Ground truth matrix
:returns: Float that contains the minimum peak area for it to be considered a spike
"""
def find_minimum_peak_area(m_ground):
    print("Finding")
    areas, amplitudes, onset_times, end_times = convolve_with_kernel(m_ground)
    sorted_areas = np.sort(areas)
    min_area = sorted_areas[np.floor(0.95*len(sorted_areas)).astype(int)]
    return min_area

"""
Find indices of significant regions of firing pattern
:param m_sample: Matrix of sample
:param minimum_peak_area: Size that a peak has to have to be considered an event
:returns: List of tuples of (begin, end) forall significant peaks
"""
def get_indices_significant_overlap(m_sample, minimum_peak_area):
    #n_samples = 1 # Number of samples of ground truth
    #l = list(map(find_minimum_peak_area, [f_ground_hypothesis(m_sample, width) for i in range(n_samples)]))
    #minimum_peak_area = sum(l)/len(l)
    areas, amplitudes, starts, ends = convolve_with_kernel(m_sample)
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
def locate_indices_neuron_per_pattern(m, minimum_peak_area):
    ind = get_indices_significant_overlap(m_sample, minimum_peak_area)
    print("Done identifying indices")
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


if __name__ == "__main__":
    #Cortex
    file_name = 'Scope1_denoised_mc_results.csv'
    weight_factor_for_detecting_SPs = 1
    m_sample = np.loadtxt(file_name, delimiter=',')
    z = locate_indices_neuron_per_pattern(m_sample, weight_factor_for_detecting_SPs)
    plt.imshow(z)
    plt.show()
    #h = np.sum(z, axis=1)
    #plt.subplot(1, 2, 1)
    #plt.bar(np.arange(0, 199), h, width=1)
    #plt.xlabel("Cell number")
    #plt.ylabel("Number of significant synchronous firing patterns partaken in")
    #plt.title("There is a large inhomogeneity of SP participation in the cortex")

    #plt.subplot(1, 2, 2)
    #plt.bar(np.arange(0, 199), sorted(h), width=1)
    #plt.xlabel("Cell number sorted")
    #plt.title("This inhomogeneity is not due to chance")

    #plt.show()
