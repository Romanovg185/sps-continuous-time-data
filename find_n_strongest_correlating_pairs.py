import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from multiprocessing import Pool
from order_by_x_coord import load_mask_to_3d_matrix, determine_ordering
import os

def generate_n_biggest(n):
    n = 10
    files = os.popen('ls SynchronousEventParticipatingNeurons/cbl_*').read().split('\n')[:-1]
    for my_file in files:
        p_cbl = np.loadtxt(my_file, delimiter=',') 
        p_ctx = np.loadtxt(my_file.replace('cbl', 'ctx'), delimiter=',') 
        p = np.vstack([p_cbl, p_ctx])
        n_cbl = p_cbl.shape[0]
        ret = []
        previous_best = []
        for h in range(10):
            d = 0
            for i, first in enumerate(p):
                for j, second in enumerate(p):
                    new_d = np.sum(np.logical_and(first, second))
                    if new_d > d and (i, j, new_d) not in previous_best and i != j:
                        d = new_d
                        best = (i, j, new_d)
            previous_best.append(best)
            if best[0] > n_cbl:
                if best[1] > n_cbl:
                    best = ('ctx{}'.format(best[0] - n_cbl), 'cbl{}'.format(best[1] - n_cbl), new_d)
                else:
                    best = ('ctx{}'.format(best[0] - n_cbl), 'cbl{}'.format(best[1]), new_d)
            else:
                if best[1] > n_cbl:
                    best = ('cbl{}'.format(best[0]), 'ctx{}'.format(best[1] - n_cbl), new_d)
                else:
                    best = ('cbl{}'.format(best[0]), 'cbl{}'.format(best[1]), new_d)

            filepath = my_file[37:].replace('_participating_neurons_', '_')
            yield best, filepath


def main():
    for indices, filename in generate_n_biggest(10):
        file_raw = '/home/romano/mep/ContinuousGlobalSynchrony/TransientsSorted/*_results_{}.csv'.format(filename[4:-4])
        m_cbl = np.loadtxt(file_raw.replace('*', 'cbl'), delimiter=',')
        m_ctx = np.loadtxt(file_raw.replace('*', 'ctx'), delimiter=',')

        # Onset times
        file_raw = '/home/romano/mep/ContinuousGlobalSynchrony/FRIOnsetTimes/*_results_{}.csv'.format(filename[4:-4])
        o_cbl = np.loadtxt(file_raw.replace('*', 'cbl'), delimiter=',')
        o_ctx = np.loadtxt(file_raw.replace('*', 'ctx'), delimiter=',')

        # Intervals of significant correlation
        file_raw = '/home/romano/mep/ContinuousGlobalSynchrony/IntervalsSignificantCorrelation/intervals_significant_correlation_{}.csv'.format(filename[4:-4])
        intervals = np.loadtxt(file_raw, delimiter=',')

        # Cells contained in significant correlation
        file_raw = '/home/romano/mep/ContinuousGlobalSynchrony/SynchronousEventParticipatingNeurons/*_participating_neurons_{}.csv'.format(filename[4:-4])
        c_cbl = np.loadtxt(file_raw.replace('*', 'cbl'), delimiter=',').T
        c_ctx = np.loadtxt(file_raw.replace('*', 'ctx'), delimiter=',').T
        first, second, weight = indices
        if first[:3] == 'cbl':
            signal_1 = m_cbl[:, int(first[3:])]
            onsets_1 = o_cbl[:, int(first[3:])]
            contained_1 = c_cbl[:, int(first[3:])]
        if first[:3] == 'ctx':
            signal_1 = m_ctx[:, int(first[3:])]
            onsets_1 = o_ctx[:, int(first[3:])]
            contained_1 = c_ctx[:, int(first[3:])]
        if second[:3] == 'cbl':
            signal_2 = m_cbl[:, int(second[3:])]
            onsets_2 = o_cbl[:, int(second[3:])]
            contained_2 = c_cbl[:, int(second[3:])]
        if second[:3] == 'ctx':
            signal_2 = m_ctx[:, int(second[3:])]
            onsets_2 = o_ctx[:, int(second[3:])]
            contained_2 = c_ctx[:, int(second[3:])]

        m = np.max(signal_1) if np.max(signal_1) > np.max(signal_2) else np.max(signal_2)
        plt.subplot(2, 1, 1)
        plt.plot(1/30*np.arange(0, len(signal_1)), signal_1)
        plt.subplot(2, 1, 2)
        plt.plot(1/30*np.arange(0, len(signal_2)), signal_2)
        plt.subplot(2, 1, 1)
        plt.scatter(onsets_1, m*np.ones_like(onsets_1), c='C0')
        plt.subplot(2, 1, 2)
        plt.scatter(onsets_2, m*np.ones_like(onsets_2), c='C0')

        firsts = intervals[:, 0]
        seconds = intervals[:, 1]
        print(contained_1)
        print(contained_2)
        lab0 = True
        lab1 = True
        lab2 = True
        lab3 = True
        for i, (first, second) in enumerate(zip(firsts, seconds)):
            if (not contained_1[i]) and (not contained_2[i]):
                if lab0:
                    plt.subplot(2, 1, 1)
                    plt.fill_between([first, second], [m, m], color=(0.05, 0.05, 0.05, 0.5), label='Both not in event')
                    plt.subplot(2, 1, 2)
                    plt.fill_between([first, second], [m, m], color=(0.05, 0.05, 0.05, 0.5), label='Both not in event')
                    lab0 = False
                else:
                    plt.subplot(2, 1, 1)
                    plt.fill_between([first, second], [m, m], color=(0.05, 0.05, 0.05, 0.5))
                    plt.subplot(2, 1, 2)
                    plt.fill_between([first, second], [m, m], color=(0.05, 0.05, 0.05, 0.5))
            if (contained_1[i]) and (not contained_2[i]):
                if lab1:
                    plt.subplot(2, 1, 1)
                    plt.fill_between([first, second], [m, m], color=(0.0, 0.0, 0.7, 0.5), label='Cell 1 in event')
                    plt.subplot(2, 1, 2)
                    plt.fill_between([first, second], [m, m], color=(0.0, 0.0, 0.7, 0.5), label='Cell 1 in event')
                    lab1 = False
                else:
                    plt.subplot(2, 1, 1)
                    plt.fill_between([first, second], [m, m], color=(0.0, 0.0, 0.7, 0.5))
                    plt.subplot(2, 1, 2)
                    plt.fill_between([first, second], [m, m], color=(0.0, 0.0, 0.7, 0.5))
            if (not contained_1[i]) and (contained_2[i]):
                if lab2:
                    plt.subplot(2, 1, 1)
                    plt.fill_between([first, second], [m, m], color=(0.7, 0.4, 0.0, 0.5), label='Cell 2 in event')
                    plt.subplot(2, 1, 2)
                    plt.fill_between([first, second], [m, m], color=(0.7, 0.4, 0.0, 0.5), label='Cell 2 in event')
                    lab2 = False
                else:
                    plt.subplot(2, 1, 1)
                    plt.fill_between([first, second], [m, m], color=(0.7, 0.4, 0.0, 0.5))
                    plt.subplot(2, 1, 2)
                    plt.fill_between([first, second], [m, m], color=(0.7, 0.4, 0.0, 0.5))
            if (contained_1[i]) and (contained_2[i]):
                if lab3:
                    plt.subplot(2, 1, 1)
                    plt.fill_between([first, second], [m, m], color=(0.7, 0.0, 0.7, 0.5), label='Both cells in event')
                    plt.subplot(2, 1, 2)
                    plt.fill_between([first, second], [m, m], color=(0.7, 0.0, 0.7, 0.5), label='Both cells in event')
                    lab3 = False
                else:
                    plt.subplot(2, 1, 1)
                    plt.fill_between([first, second], [m, m], color=(0.7, 0.0, 0.7, 0.5))
                    plt.subplot(2, 1, 2)
                    plt.fill_between([first, second], [m, m], color=(0.7, 0.0, 0.7, 0.5))
        plt.subplot(2, 1, 1)
        plt.legend()
        plt.xlim([100, 137])
        plt.ylabel(r'$\frac{\Delta F}{F}$')
        plt.title(filename[4:] + ' (' + indices[0][:3] + ', ' + indices[1][:3] + ')')
        plt.subplot(2, 1, 2)
        plt.xlim([100, 137])
        plt.ylabel(r'$\frac{\Delta F}{F}$')
        plt.xlabel('Time [sec]')
        plt.show()

if __name__ == '__main__':
    main()
