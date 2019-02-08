import numpy as np
import matplotlib.pyplot as plt


def make_ground_hypothesis(m):
    itis = []
    for i in m.T:
        spikes = i[np.logical_not(np.isnan(i))]
        iti = spikes[1:] - spikes[:-1]
        itis.extend(iti)
    gram = sorted(itis) - min(itis)
    lambda_hat = 1/np.mean(gram) # Maximum likelihood estimator of exponential distribution is 1/lambda as per Wiki

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

def convolve_with_kernel(m, kernel_steepness=0.020, c='b'):
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

    peak_areas = [0.001*sum(i) for i in peaks]
    times_peak_starts = [0.001*(i-1) for i in indices_peak_starts]
    times_peak_ends = [0.001*(i-1) for i in indices_peak_ends]
    peak_amplitudes = [np.max(i) for i in peaks]
    plt.hist(peak_amplitudes, alpha=0.5)

m = np.loadtxt('/home/romano/mep/TemporalHyperaccuity/hyperaccuity_output/150421-Bl01_package.csv', delimiter=',')
mprime = make_ground_hypothesis(m)
convolve_with_kernel(m, c='C0')
convolve_with_kernel(mprime, c='C1')
plt.xlabel('Maximum fluorescence [Fluorescence]')
plt.ylabel('Occurences')
plt.show()
