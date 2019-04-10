import numpy as np
import matplotlib.pyplot as plt

def epac(m):
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
        print('{}/{}'.format(i+1, m.shape[1]))
    z = np.sum(y, axis=0)
    return z

def plot_groundness_truth(m_raw, n=10):
    for _ in range(n):
        d_log_cells = []
        max_t = 0
        for i, cell in enumerate(m_raw.T):
            if np.max(cell) > max_t:
                max_t = np.max(cell)
            d_cell = cell[1:] - cell[:-1]
            d_log_cell = np.log(d_cell)
            d_log_cells.extend(i for i in d_log_cell if not np.isnan(i))
        mu = np.mean(d_log_cells)
        sd = np.std(d_log_cells)
        m_sim = np.full((1000, m_raw.shape[1]), np.nan)
        for i in range(m_raw.shape[1]):
            sim_d_log = np.random.normal(mu, sd, 1000)
            sim_d = np.exp(sim_d_log)
            sim_cell = np.cumsum(sim_d)
            sim_cell = sim_cell[sim_cell < max_t]
            m_sim[:len(sim_cell), i] = sim_cell
        mask = ~np.isnan(np.nanmean(m_sim, axis=1))
        m_sim = m_sim[mask, :]
        plt.subplot(1, 2, 1)
        for i, t0s in enumerate(m_sim.T):
            plt.scatter(t0s, i*np.ones_like(t0s), s=5)
        z = epac(m_sim)
        z_sampled = z[50::100]
        t_i = np.arange(0, len(z_sampled))
        t_f = np.arange(1, len(z_sampled)+1)
        s = int(m_raw.shape[1])
        for z, i, f in zip(z_sampled, t_i, t_f):
            plt.fill_between([i, f], [0, 0], [s, s], facecolor=(1 - z/10, 1-z/10, 1-z/10, 0.3))
        plt.xlim([0, 20])
        plt.xlabel('Time [sec]')
        plt.ylabel('Cell id')
        plt.title('Surrogate data')

        plt.subplot(1, 2, 2)
        for i, t0s in enumerate(m_raw.T):
            plt.scatter(t0s, i*np.ones_like(t0s), s=5)
        z = epac(m_raw)
        z_sampled = z[50::100]
        t_i = np.arange(0, len(z_sampled))
        t_f = np.arange(1, len(z_sampled)+1)
        s = int(m_raw.shape[1])
        for z, i, f in zip(z_sampled, t_i, t_f):
            plt.fill_between([i, f], [0, 0], [s, s], facecolor=(1 - z/10, 1-z/10, 1-z/10, 0.3))
        plt.xlim([0, 20])
        plt.xlabel('Time [sec]')
        plt.ylabel('Cell id')
        plt.title('Real data')

        plt.show()


def generate_ground_truth(m_raw, n=10):
    for _ in range(n):
        d_log_cells = []
        max_t = 0
        for i, cell in enumerate(m_raw.T):
            if np.max(cell) > max_t:
                max_t = np.max(cell)
            d_cell = cell[1:] - cell[:-1]
            d_log_cell = np.log(d_cell)
            d_log_cells.extend(i for i in d_log_cell if not np.isnan(i))
        mu = np.mean(d_log_cells)
        sd = np.std(d_log_cells)
        m_sim = np.full((1000, m_raw.shape[1]), np.nan)
        for i in range(m_raw.shape[1]):
            sim_d_log = np.random.normal(mu, sd, 1000)
            sim_d = np.exp(sim_d_log)
            sim_cell = np.cumsum(sim_d)
            sim_cell = sim_cell[sim_cell < max_t]
            m_sim[:len(sim_cell), i] = sim_cell
        mask = ~np.isnan(np.nanmean(m_sim, axis=1))
        m_sim = m_sim[mask, :]
        yield m_sim

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
        print('{}/{}'.format(i+1, m.shape[1]))
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


def plot_kernel_sum_comparisons():
    m_real = np.loadtxt('/home/romano/mep/ContinuousGlobalSynchrony/FRIOnsetTimes/cbl_results_16082018_120304.csv', delimiter=',')
    for h, m in enumerate(generate_ground_truth(m_real, n=3)):
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
            print('{}/{}'.format(i+1, m.shape[1]))
        z = np.sum(y, axis=0)
        plt.plot(t[0, :], z, label='Ground truth instance {}'.format(h))

    m = m_real
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
        print('{}/{}'.format(i+1, m.shape[1]))
    z = np.sum(y, axis=0)
    plt.plot(t[0, :], z, c='k', label='Real kernel sum')
    plt.legend()
    plt.xlabel('Time [sec]')
    plt.ylabel('Value kernel sum')
    plt.show()

if __name__ == "__main__":
    m_real = np.loadtxt('/home/romano/mep/ContinuousGlobalSynchrony/FRIOnsetTimes/cbl_results_16082018_120304.csv', delimiter=',')
    plot_groundness_truth(m_real, n=1)
