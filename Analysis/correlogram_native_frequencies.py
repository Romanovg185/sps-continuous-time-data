import numpy as np
import matplotlib.pyplot as plt
import os

def paired_mean_frequency(filename_cerebellum, filename_cortex):
    cbl_t0 = np.loadtxt('/home/romano/Documents/ContinuousGlobalSynchrony/FRIOnsetTimes/cbl_results_{}'.format(filename_cerebellum[4:]), delimiter=',')
    ctx_t0 = np.loadtxt('/home/romano/Documents/ContinuousGlobalSynchrony/FRIOnsetTimes/ctx_results_{}'.format(filename_cortex[4:]), delimiter=',')
    t_max = np.nanmax(cbl_t0) if np.nanmax(cbl_t0) > np.nanmax(ctx_t0) else np.nanmax(ctx_t0)
    cbl_f = (np.nansum(cbl_t0, axis=0)/t_max).reshape(-1, 1)
    ctx_f = (np.nansum(ctx_t0, axis=0)/t_max).reshape(-1, 1)
    n_cbl = cbl_t0.shape[1]
    n_ctx = ctx_t0.shape[1]
    n = n_cbl + n_ctx
    freqs = np.zeros((n, n))
    freqs[:n_cbl, :n_cbl] = (cbl_f + cbl_f.T)/2
    freqs[n_cbl:, n_cbl:] = (ctx_f + ctx_f.T)/2
    freqs[:n_cbl, n_cbl:] = (cbl_f + ctx_f.T)/2
    freqs[n_cbl:, :n_cbl] = (ctx_f + cbl_f.T)/2

    return None, freqs, n_cbl/n, cbl_t0.shape[1]

def export_four_box_plots():
    files = os.popen('ls SynchronousEventParticipatingNeurons').read().split('\n')[:-1]
    files = list({i[3:] for i in files})
    cortex_files = ['ctx' + i for i in files]
    cerebellum_files = ['cbl' + i for i in files]
    for file_name_cerebellum, file_name_cortex in zip(cerebellum_files, cortex_files):
        z_tot, correlation_score, ratio, scope1_valid = paired_mean_frequency(file_name_cerebellum, file_name_cortex)
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
        plt.savefig('FreqBoxPlots/' + file_name_cerebellum[4:-3] + 'eps')
        #np.savetxt('four_box_data' + file_name_cerebellum[24:], z_tot, delimiter=',')

export_four_box_plots()
