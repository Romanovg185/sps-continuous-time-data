import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm

def number_of_pairs_per_threshold():
    
    partaking_neurons_per_event_scope1 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/Data/Partaking_Cells_Per_Synchronous_Event_Scope1.csv', delimiter=',')    
    partaking_neurons_per_event_scope2 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/Data/Partaking_Cells_Per_Synchronous_Event_Scope2.csv', delimiter=',')    
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

def correlation_score_matrix_above_threshold():
    is_plotting_autocorrelation = False
    is_logarithmic_color_map = False
    
    partaking_neurons_per_event_scope1 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/Data/Partaking_Cells_Per_Synchronous_Event_Scope1.csv', delimiter=',')    
    partaking_neurons_per_event_scope2 = np.loadtxt('/home/romano/Documents/continous-time-sp-detection/Data/Partaking_Cells_Per_Synchronous_Event_Scope2.csv', delimiter=',')    
    total_partaking = np.vstack([partaking_neurons_per_event_scope1, partaking_neurons_per_event_scope2]).astype(bool)
    n_scope_1 = partaking_neurons_per_event_scope1.shape[0]
    
    size = partaking_neurons_per_event_scope1.shape[0] + partaking_neurons_per_event_scope2.shape[0]
    ratio = n_scope_1/size
    correlation_score = np.zeros((size, size))

    threshold = 10
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

    plt.show()

if __name__ == "__main__":
    correlation_score_matrix_above_threshold()
