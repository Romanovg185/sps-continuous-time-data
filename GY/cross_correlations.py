import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.io import loadmat

def read():
    z = loadmat('/home/romano/mep/BrianDataAnalyze/Data/sorted_masks_cortex_cerebellum_left_to_right.mat')
    A_cerebellum = z['A_cerebellum']
    A_cortex = z['A_cortex']
    sortedx_cerebellum = z['sortedx_cerebellum']
    sortedx_cortex = z['sortedx_cortex']
    print(A_cerebellum.shape)
    print(A_cortex.shape)
    print(sortedx_cerebellum.shape)
    print(sortedx_cortex.shape)
    print(sortedx_cortex)
    plt.plot(sortedx_cortex[:, 1], sortedx_cortex[:, 0])
    print([i <= j for i, j in zip(sortedx_cerebellum[:, 0][:-1], sortedx_cerebellum[:, 0][1:])])
    plt.show()


def main():
    scope1_sig = np.loadtxt('/home/romano/mep/BrianDataAnalyze/Data/Patterns_Scope1.csv', delimiter=',')
    scope2_sig = np.loadtxt('/home/romano/mep/BrianDataAnalyze/Data/Patterns_Scope2.csv', delimiter=',')
    m = np.zeros((scope1_sig.shape[0], scope2_sig.shape[0]))
    print(m.shape)
    for i, cell_1 in enumerate(scope1_sig):
        print(i)
        for j, cell_2 in enumerate(scope2_sig):
            m[i, j] = np.sum(correlate(cell_1, cell_2))
    print(np.where(m == np.max(m)))
    plt.imshow(m)
    plt.xlabel('Cortex cell number')
    plt.ylabel('Cerebellum cell number')
    plt.colorbar()
    plt.show()

    maximum_correlation_indices = (210, 258)
    plt.plot(1/30*np.arange(scope1_sig.shape[1]), scope1_sig[210, :])
    plt.plot(1/30*np.arange(scope2_sig.shape[1]), scope2_sig[258, :])
    plt.show()

if __name__ == '__main__':
    main()
