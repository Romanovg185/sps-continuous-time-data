import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
import os

"""
Plots the pairwise shared number of significant correlation events per neuron pair.
A neuron (column/row) is filtered out if there is no other neuron with which it shares more than threshold firing events
:param threshold: Minimum number of shared events between a neuron and an arbitrary other neuron to not be filtered out
"""
def correlation_score_matrix_above_threshold(path_ctx, threshold=0):
    is_plotting_autocorrelation = False
    
    partaking_neurons_per_event_scope1 = np.loadtxt('/home/romano/Documents/ContinuousGlobalSynchrony/SynchronousEventParticipatingNeurons/{}'.format(path_ctx).replace("ctx", "cbl"), delimiter=',')    
    partaking_neurons_per_event_scope2 = np.loadtxt('/home/romano/Documents/ContinuousGlobalSynchrony/SynchronousEventParticipatingNeurons/{}'.format(path_ctx), delimiter=',')    
    total_partaking = np.vstack([partaking_neurons_per_event_scope1, partaking_neurons_per_event_scope2]).astype(bool)
    print(total_partaking.shape)
    n_scope_1 = partaking_neurons_per_event_scope1.shape[0]
    
    size_cbl = partaking_neurons_per_event_scope1.shape[0]
    size = partaking_neurons_per_event_scope1.shape[0] + partaking_neurons_per_event_scope2.shape[0]
    ratio = n_scope_1/size
    correlation_score = np.zeros((size, size))

    for i, first in enumerate(total_partaking):
        for j, sec in enumerate(total_partaking):
            events_shared = np.sum(np.logical_and(first, sec))
            if events_shared >= threshold:
                if is_plotting_autocorrelation or i != j:
                    correlation_score[i, j] = events_shared
    
    scope1_valid = size_cbl # Number of true columns in the cerebellum

    # Writing
    z_top = np.hstack([correlation_score[:scope1_valid, :scope1_valid], np.full((scope1_valid, 1), np.nan), correlation_score[:scope1_valid, scope1_valid:]])
    z_bot = np.hstack([correlation_score[scope1_valid:, :scope1_valid], np.full((correlation_score.shape[0] - scope1_valid, 1), np.nan), correlation_score[scope1_valid:, scope1_valid:]])
    z_tot = np.vstack([z_top, np.full((1, correlation_score.shape[0] + 1), np.nan), z_bot])
    return z_tot, correlation_score, ratio, scope1_valid

def export_four_box_plots():
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
        print(file_name_cerebellum)
        plt.savefig('FourBoxPlots/' + file_name_cerebellum[4:-3] + 'eps')
        np.savetxt('FourBoxRawData/' + file_name_cerebellum[4:], z_tot, delimiter=',')

if __name__ == "__main__":
    export_four_box_plots()
    #print(r)
