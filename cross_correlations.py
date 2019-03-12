import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

def main():
    scope1_sig = np.loadtxt('/home/romano/mep/BrianDataAnalyze/Data/Patterns_Scope1.csv', delimiter=',')
    scope2_sig = np.loadtxt('/home/romano/mep/BrianDataAnalyze/Data/Patterns_Scope2.csv', delimiter=',')
    m = np.zeros((scope1_sig.shape[0], scope2_sig.shape[0]))
    print(m.shape)
    for i, cell_1 in enumerate(scope1_sig):
        print(i)
        for j, cell_2 in enumerate(scope2_sig):
            m[i, j] = np.sum(correlate(cell_1, cell_2))
    plt.imshow(m)
    plt.xlabel('Cerebellum cell number')
    plt.ylabel('Cortex cell number')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()
