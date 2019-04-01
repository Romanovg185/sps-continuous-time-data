import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
import os

"""
Plots a line that shows the number of shared events per neuron pair
"""
def number_of_pairs_per_threshold(path_ctx):
    partaking_neurons_per_event_scope1 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/SynchronousEventParticipatingNeurons/{}'.format(path_ctx), delimiter=',')    
    partaking_neurons_per_event_scope2 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/SynchronousEventParticipatingNeurons/{}'.format(path_ctx).replace("ctx", "cbl"), delimiter=',')    
    total_partaking = np.vstack([partaking_neurons_per_event_scope1, partaking_neurons_per_event_scope2]).astype(bool)
    n_scope_1 = partaking_neurons_per_event_scope1.shape[0]
    
    scope11 = []
    scope22 = []
    scope12 = []
    for h in range(74):
        print(h)
        threshold = h
        significant_correlated_cells = []
        for i, first in enumerate(total_partaking):
            for j, sec in enumerate(total_partaking[i+1:]):
                events_shared = np.sum(np.logical_and(first, sec))
                if events_shared >= threshold:
                    significant_correlated_cells.append((i,j))
        scope11.append(len([i for i in significant_correlated_cells if (i[0] < n_scope_1 and i[1] < n_scope_1)]))
        scope22.append(len([i for i in significant_correlated_cells if (i[0] >= n_scope_1 and i[1] >= n_scope_1)]))
        scope12.append(len(significant_correlated_cells) - scope11[-1] - scope22[-1])
    plt.semilogy([h for h in range(74)], scope11, label='Both cerebellar')
    plt.semilogy([h for h in range(74)], scope22, label='Both cortical')
    plt.semilogy([h for h in range(74)], scope12, label='Different regions')
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Number of significant cell pairs')
    plt.show()

"""
Plots the pairwise shared number of significant correlation events per neuron pair
"""
def correlation_score_matrix():
    is_plotting_autocorrelation = True
    is_logarithmic_color_map = True
    
    partaking_neurons_per_event_scope1 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/Data/Partaking_Cells_Per_Synchronous_Event_Scope1.csv', delimiter=',')    
    partaking_neurons_per_event_scope2 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/Data/Partaking_Cells_Per_Synchronous_Event_Scope2.csv', delimiter=',')    
    total_partaking = np.vstack([partaking_neurons_per_event_scope1, partaking_neurons_per_event_scope2]).astype(bool)
    n_scope_1 = partaking_neurons_per_event_scope1.shape[0]
    
    size = partaking_neurons_per_event_scope1.shape[0] + partaking_neurons_per_event_scope2.shape[0]
    ratio = n_scope_1/size
    correlation_score = np.zeros((size, size))

    threshold = 0
    for i, first in enumerate(total_partaking):
        for j, sec in enumerate(total_partaking):
            events_shared = np.sum(np.logical_and(first, sec))
            if events_shared >= threshold:
                if is_plotting_autocorrelation or i != j:
                    correlation_score[i, j] = events_shared
    if is_logarithmic_color_map:
        correlation_score = np.log(correlation_score)
        current_cmap = cm.get_cmap()
        current_cmap.set_bad(color='black')
    f, axes = plt.subplots(2, 2, sharey='row', sharex='col', gridspec_kw={'width_ratios': [ratio, 1-ratio], 'height_ratios': [ratio, 1-ratio]})
    axes[0, 0].imshow(correlation_score[:n_scope_1, :n_scope_1], aspect='auto')
    axes[0, 0].xaxis.tick_top()
    axes[0, 0].set_ylabel('Cerebellum')
    axes[0, 0].set_xlabel('Cerebellum')
    axes[0, 0].xaxis.set_label_position('top')

    axes[0, 1].imshow(correlation_score[:n_scope_1, n_scope_1:], aspect='auto')
    axes[0, 1].xaxis.tick_top()
    axes[0, 1].set_xlabel('Cortex')
    axes[0, 1].xaxis.set_label_position('top')

    axes[1, 0].imshow(correlation_score[n_scope_1:, :n_scope_1], aspect='auto')
    axes[1, 0].set_ylabel('Cortex')

    axes[1, 1].imshow(correlation_score[n_scope_1:, n_scope_1:], aspect='auto')

    plt.show()

"""
Plots the pairwise shared number of significant correlation events per neuron pair.
A neuron (column/row) is filtered out if there is no other neuron with which it shares more than threshold firing events
:param threshold: Minimum number of shared events between a neuron and an arbitrary other neuron to not be filtered out
"""
def correlation_score_matrix_above_threshold(path_ctx, threshold=1):
    is_plotting_autocorrelation = False
    is_logarithmic_color_map = False
    
    partaking_neurons_per_event_scope1 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/SynchronousEventParticipatingNeurons/{}'.format(path_ctx), delimiter=',')    
    partaking_neurons_per_event_scope2 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/SynchronousEventParticipatingNeurons/{}'.format(path_ctx).replace("ctx", "cbl"), delimiter=',')    
    total_partaking = np.vstack([partaking_neurons_per_event_scope1, partaking_neurons_per_event_scope2]).astype(bool)
    n_scope_1 = partaking_neurons_per_event_scope1.shape[0]
    
    size = partaking_neurons_per_event_scope1.shape[0] + partaking_neurons_per_event_scope2.shape[0]
    ratio = n_scope_1/size
    correlation_score = np.zeros((size, size))

    for i, first in enumerate(total_partaking):
        for j, sec in enumerate(total_partaking):
            events_shared = np.sum(np.logical_and(first, sec))
            if events_shared >= threshold:
                if is_plotting_autocorrelation or i != j:
                    correlation_score[i, j] = events_shared
    if is_logarithmic_color_map:
        correlation_score = np.log(correlation_score)
        current_cmap = cm.get_cmap()
        current_cmap.set_bad(color='black')
    
    sum_of_correlations = np.sum(correlation_score[:n_scope_1], axis=1)
    scope1_valid = np.sum(sum_of_correlations > 0) # Number of true columns in the cerebellum
    correlation_score = correlation_score[~np.all(correlation_score == 0, axis=1)]
    correlation_score = correlation_score[:, ~np.all(correlation_score == 0, axis=0)]

    # Writing
    z_top = np.hstack([correlation_score[:scope1_valid, :scope1_valid], np.full((scope1_valid, 1), np.nan), correlation_score[:scope1_valid, scope1_valid:]])
    z_bot = np.hstack([correlation_score[scope1_valid:, :scope1_valid], np.full((correlation_score.shape[0] - scope1_valid, 1), np.nan), correlation_score[scope1_valid:, scope1_valid:]])
    z_tot = np.vstack([z_top, np.full((1, correlation_score.shape[0] + 1), np.nan), z_bot])
    return z_tot, correlation_score, ratio, scope1_valid
    

"""
Plots the pairwise shared number of significant correlation events per neuron pair.
A neuron (column/row) is filtered out if there is no other neuron with which it shares more than threshold firing events
:param threshold: Minimum number of shared events between a neuron and an arbitrary other neuron to not be filtered out
:returns: List of [((cell, cell), correlation_score)] sorted by correlation score
"""
def sorted_correlation_score_matrix_above_threshold(path_ctx, threshold=1):
    is_plotting_autocorrelation = False
    is_logarithmic_color_map = False
    partaking_neurons_per_event_scope1 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/SynchronousEventParticipatingNeurons/{}'.format(path_ctx), delimiter=',')    
    partaking_neurons_per_event_scope2 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/SynchronousEventParticipatingNeurons/{}'.format(path_ctx).replace("ctx", "cbl"), delimiter=',')    
    total_partaking = np.vstack([partaking_neurons_per_event_scope1, partaking_neurons_per_event_scope2]).astype(bool)
    n_scope_1 = partaking_neurons_per_event_scope1.shape[0]
    
    size = partaking_neurons_per_event_scope1.shape[0] + partaking_neurons_per_event_scope2.shape[0]
    ratio = n_scope_1/size
    correlation_score = np.zeros((size, size))

    for i, first in enumerate(total_partaking):
        for j, sec in enumerate(total_partaking):
            events_shared = np.sum(np.logical_and(first, sec))
            if events_shared >= threshold:
                if is_plotting_autocorrelation or i != j:
                    correlation_score[i, j] = events_shared
    if is_logarithmic_color_map:
        correlation_score = np.log(correlation_score)
        current_cmap = cm.get_cmap()
        current_cmap.set_bad(color='black')

    # Obtaining list of cells sorted by correlation score
    correlation_score_copy = np.copy(correlation_score) # Copy to not disrupt code downstream
    ret = []
    while True:
        max_correlation_score = np.max(correlation_score_copy)
        print(max_correlation_score)
        if max_correlation_score < threshold:
            break
        mask = correlation_score_copy == max_correlation_score*np.ones_like(correlation_score_copy)
        a = np.nonzero(mask)
        ind = (a[0][0], a[1][0])
        print(ind)
        print('.')
        ret.append((ind, max_correlation_score))
        correlation_score_copy[ind] = 0

    correlation_sum = np.sum(correlation_score, axis=1)
    inds = [i[0] for i in sorted(zip(np.arange(0, len(correlation_sum)), correlation_sum), key=lambda x: x[1], reverse=True)]
    l = [i for i in sorted(zip(correlation_score, correlation_sum), key=lambda x: x[1], reverse=True) if i[1] > 0]
    inds = inds[:len(l)]
    m = np.vstack([i[0] for i in l])
    f, axes = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [ratio, 1-ratio]})
    axes[0].imshow(m[:, :n_scope_1].T, aspect='auto')
    axes[0].set_ylabel('Cerebellum cell number')
    axes[1].imshow(m[:, n_scope_1:].T, aspect='auto')
    axes[1].set_ylabel('Cortex cell number')
    axes[1].set_xlabel('Cell number above threshold')

    axes[1].set_xticks(np.arange(0, m.shape[0]))
    axes[1].set_xticklabels(inds, rotation=45)
    plt.xlim([0, 50])
    plt.show()
    return ret



"""
Plots the pairwise shared number of significant correlation events per neuron pair.
Cerebellar and cortical cells are split from each other, returning four plots
A neuron (column/row) is filtered out if there is no other neuron with which it shares more than threshold firing events
:param threshold: Minimum number of shared events between a neuron and an arbitrary other neuron to not be filtered out
:returns: List of [((cell, cell), correlation_score)] sorted by correlation score
"""
def double_sorted_correlation_score_matrix_above_threshold(path_ctx, threshold=1):
    is_plotting_autocorrelation = False
    is_logarithmic_color_map = False
    
    partaking_neurons_per_event_scope1 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/SynchronousEventParticipatingNeurons/{}'.format(path_ctx), delimiter=',')    
    partaking_neurons_per_event_scope2 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/SynchronousEventParticipatingNeurons/{}'.format(path_ctx).replace("ctx", "cbl"), delimiter=',')    
    total_partaking = np.vstack([partaking_neurons_per_event_scope1, partaking_neurons_per_event_scope2]).astype(bool)
    n_scope_1 = partaking_neurons_per_event_scope1.shape[0]
    
    size = partaking_neurons_per_event_scope1.shape[0] + partaking_neurons_per_event_scope2.shape[0]
    ratio = n_scope_1/size
    correlation_score = np.zeros((size, size))

    for i, first in enumerate(total_partaking):
        for j, sec in enumerate(total_partaking):
            events_shared = np.sum(np.logical_and(first, sec))
            if events_shared >= threshold:
                if is_plotting_autocorrelation or i != j:
                    correlation_score[i, j] = events_shared
    if is_logarithmic_color_map:
        correlation_score = np.log(correlation_score)
        current_cmap = cm.get_cmap()
        current_cmap.set_bad(color='black')

    # Obtaining list of cells sorted by correlation score
    correlation_score_copy = np.copy(correlation_score) # Copy to not disrupt code downstream
    ret = []
    while True:
        max_correlation_score = np.max(correlation_score_copy)
        print(max_correlation_score)
        if max_correlation_score < threshold:
            break
        mask = correlation_score_copy == max_correlation_score*np.ones_like(correlation_score_copy)
        a = np.nonzero(mask)
        ind = (a[0][0], a[1][0])
        print(ind)
        print('.')
        ret.append((ind, max_correlation_score))
        correlation_score_copy[ind] = 0

    correlation_sum = np.sum(correlation_score, axis=1)
    print(correlation_sum)
    inds = [i[0] for i in sorted(zip(np.arange(0, len(correlation_sum)), correlation_sum), key=lambda x: x[1], reverse=True)]
    print(inds)
    l = [i for i in sorted(zip(correlation_score, correlation_sum), key=lambda x: x[1], reverse=True) if i[1] > 0]
    inds = inds[:len(l)]
    is_in_scope1 = np.array(inds) > n_scope_1
    m = np.vstack([i[0] for i in l])

    m_scope1 = m[is_in_scope1]
    m_scope2 = m[~is_in_scope1]
    inds_scope1 = [i for i in inds if i < n_scope_1]
    inds_scope2 = [i - n_scope_1 for i in inds if i >= n_scope_1]

    # Plotting
    f, axes = plt.subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': [ratio, 1-ratio]})
    axes[0, 0].imshow(m_scope1[:, :n_scope_1].T, aspect='auto')
    axes[0, 0].set_ylabel('Cerebellum cell number')

    axes[0, 1].imshow(m_scope2[:, :n_scope_1].T, aspect='auto')

    axes[1, 0].imshow(m_scope1[:, n_scope_1:].T, aspect='auto')
    axes[1, 0].set_xticks(np.arange(0, len(inds_scope1)))
    axes[1, 0].set_xticklabels(inds_scope1, rotation=45)
    axes[1, 0].set_ylabel('Cortical cell number')
    axes[1, 0].set_xlabel('Correlating cerebellar cell number')

    axes[1, 1].imshow(m_scope2[:, n_scope_1:].T, aspect='auto')
    axes[1, 1].set_xticks(np.arange(0, len(inds_scope2)))
    axes[1, 1].set_xticklabels(inds_scope2, rotation=45)
    axes[1, 1].set_xlabel('Correlating cortical cell number')

    axes[0, 0].set_xlim([0, 25])
    plt.xlim([0, 25])
    plt.show()
    
    return ret

if __name__ == "__main__":
    files = os.popen('ls SynchronousEventParticipatingNeurons').read().split('\n')[:-1]
    files = list({i[3:] for i in files})
    cortex_files = ['ctx' + i for i in files]
    cerebellum_files = ['cbl' + i for i in files]
    for file_name_cerebellum, file_name_cortex in zip(cerebellum_files, cortex_files):
        z_tot, correlation_score, ratio, scope1_valid = correlation_score_matrix_above_threshold(file_name_cortex)
        f, axes = plt.subplots(2, 2, sharey='row', sharex='col', gridspec_kw={'width_ratios': [ratio, 1-ratio], 'height_ratios': [ratio, 1-ratio]})
        m = correlation_score[:scope1_valid, :scope1_valid]
        axes[0, 0].imshow(m, aspect='auto')
        axes[0, 0].xaxis.tick_top()
        axes[0, 0].set_ylabel('Cerebellum')
        axes[0, 0].set_xlabel('Cerebellum')
        axes[0, 0].xaxis.set_label_position('top')

        m = correlation_score[:scope1_valid, scope1_valid:]
        axes[0, 1].imshow(m, aspect='auto')
        axes[0, 1].xaxis.tick_top()
        axes[0, 1].set_xlabel('Cortex')
        axes[0, 1].xaxis.set_label_position('top')
        
        m = correlation_score[scope1_valid:, :scope1_valid]
        axes[1, 0].imshow(m, aspect='auto')
        axes[1, 0].set_ylabel('Cortex')

        m = correlation_score[scope1_valid:, scope1_valid:]
        axes[1, 1].imshow(m, aspect='auto')
        plt.savefig('four_box_plot' + file_name_cerebellum[24:-3] + '.eps')
        np.savetxt('four_box_data' + file_name_cerebellum[24:], z_tot, delimiter=',')

