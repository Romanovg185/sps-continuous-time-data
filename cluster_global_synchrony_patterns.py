import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from multiprocessing import Pool
from convolve_with_kernel import identify_patterns_two_samples
from scipy.cluster.hierarchy import linkage, dendrogram


if __name__ == "__main__":
    file_name_cerebellum = 'Scope1_denoised_mc_results.csv'
    file_name_cortex = 'Scope2_denoised_mc_results.csv'
    shifts = [-20, -1, -0.1, 0, 0.1, 1, 20]

    m_cbl = np.loadtxt(file_name_cerebellum, delimiter=',')
    m_ctx = np.loadtxt(file_name_cortex, delimiter=',')
    l = [(m_cbl, m_ctx + shift) for shift in shifts]
    def helper(x): z=identify_patterns_two_samples(x[0], x[1]); return np.vstack(z)
    ll = list(map(helper, l))
    fs = [np.sum(i, axis=1) for i in ll]
    for i, f in enumerate(fs):
        plt.subplot(1, 7, i+1)
        plt.hist(f, bins=[i-0.5 for i in range(int(max(f) + 1))])
        plt.title(r"$\varphi = {}$".format(shifts[i]) + r"$\: \mathrm{sec}$")
        plt.xlabel(r'$N_{\mathrm{partaking}}$')
        plt.xlim([0, 50])
    plt.show()
    clusterings = [linkage(i.reshape(-1, 1), method='average', metric='euclidean') for i in fs]
    for instance in clusterings:
        dendrogram(instance)
        plt.show()

