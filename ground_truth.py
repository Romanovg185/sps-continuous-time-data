import numpy as np
import matplotlib.pyplot as plt

### Free parameters ###
kernel_steepness = 0.100

"""
Individual kerneling
"""
def epac(cell):
    u = np.arange(-1*kernel_steepness, kernel_steepness + 0.001, 0.001)
    kernel = 4*kernel_steepness/3*(1 - (u/kernel_steepness)**2)
    t_axis = np.arange(0, 1000, 0.001)
    t = np.zeros((len(t_axis)))
    y = np.zeros_like(t)
    t += t_axis
    indices_of_spike = (cell/0.001).astype(int)
    indices_of_spike_nonan = indices_of_spike[indices_of_spike >= 0]
    y[indices_of_spike_nonan] = 1
    y = np.convolve(y, kernel, mode='same')
    return y

"""
Kernel sum
"""
def epac2(m):
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

def generate_ground_truth(m_raw, n=10): # m_raw of form [events, cells]
    for _ in range(n):
        m_ground = np.full_like(m_raw, np.nan)
        for h, i in enumerate(np.random.randint(0, m_raw.shape[1], m_raw.shape[1])):
            print('{}/{}'.format(h+1, m_raw.shape[1]))
            cell = m_raw[:, i]
            cell = cell[~np.isnan(cell)]
            n = len(cell)
            max_t = np.max(cell)
            kerneled_cell = epac(cell)[:int(1/0.001*(max_t+5*kernel_steepness))] # Have to make sure the entire final kernel fits on the screen
            autocorr = np.correlate(kerneled_cell, kerneled_cell, mode='same')
            normalized_autocorr = 1/np.sum(autocorr)*autocorr
            cum_distr = np.cumsum(normalized_autocorr)
            random_val = np.random.rand(n)
            times = [] # Firing times of surrogate data
            for val in random_val:
                ds = np.abs(val - cum_distr)
                ind = np.where(ds == np.min(ds))[0][0]
                time = ind*0.001
                times.append(time)
            times = np.array(times)
            m_ground[:len(times), h] = times 
        yield m_ground

def plot_groundness_truth(m_raw, n=10):
    for _ in range(n):
        m_sim = next(generate_ground_truth(m_raw, n=1))
        plt.subplot(1, 2, 1)
        for i, t0s in enumerate(m_sim.T):
            plt.scatter(t0s, i*np.ones_like(t0s), s=5)
        z = epac2(m_sim)
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
        z = epac2(m_raw)
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

def plot_kernel_sum_comparisons():
    m_real = np.loadtxt('/home/romano/mep/ContinuousGlobalSynchrony/FRIOnsetTimes/cbl_results_16082018_120304.csv', delimiter=',')
    zs = []
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
        z = np.sum(y, axis=0)
        zs.append(z)
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
    z = np.sum(y, axis=0)
    plt.plot(t[0, :], z, c='k', label='Real kernel sum')

    all_kernels = np.vstack(zs)
    mean_kernel = np.mean(all_kernels, axis=1)
    mu = np.mean(mean_kernel)
    sigma = np.std(mean_kernel)
    plt.plot(t_axis, mu*np.ones_like(t_axis), c='k', ls='--', label='Mean')
    plt.plot(t_axis, (mu + 2*sigma)*np.ones_like(t_axis), c='k', ls=':', label=r'$2\sigma$ above the mean')

    plt.legend()
    plt.xlabel('Time [sec]')
    plt.ylabel('Value kernel sum')
    plt.show()

if __name__ == "__main__":
    plot_kernel_sum_comparisons()
