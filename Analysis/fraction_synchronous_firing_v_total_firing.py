import numpy as np
import matplotlib.pyplot as plt

def get_native_firing_frequency_per_cell(filename=''):
    ctx_onset_times = np.loadtxt('ctx_results{}.csv'.format(filename), delimiter=',')
    cbl_onset_times = np.loadtxt('cbl_results{}.csv'.format(filename), delimiter=',')
    t_max_ctx = np.nanmax(ctx_onset_times)
    t_max_cbl = np.nanmax(cbl_onset_times)
    t_max = t_max_ctx if t_max_ctx > t_max_cbl else t_max_cbl
    mask_ctx = ~np.isnan(ctx_onset_times)
    mask_cbl = ~np.isnan(cbl_onset_times)
    f_ctx = np.sum(mask_ctx, axis=0)/t_max
    f_cbl = np.sum(mask_cbl, axis=0)/t_max
    return t_max, f_ctx, f_cbl

def get_synchronous_firing_frequency_per_cell(t_max, filename=''):
    ctx_participants = np.loadtxt('ctx_participating_neurons{}.csv'.format(filename), delimiter=',')
    cbl_participants = np.loadtxt('cbl_participating_neurons{}.csv'.format(filename), delimiter=',')
    fstar_ctx = np.sum(ctx_participants, axis=1)/t_max
    fstar_cbl = np.sum(cbl_participants, axis=1)/t_max
    return fstar_ctx, fstar_cbl
    


def plot_native_and_synchronous_firing():
    t_max, f_ctx, f_cbl = get_native_firing_frequency_per_cell()
    fstar_ctx, fstar_cbl = get_synchronous_firing_frequency_per_cell(t_max)
    ratio = len(f_ctx)/(len(f_ctx) + len(f_cbl))
    f, axes = plt.subplots(1, 2, sharey='row', gridspec_kw={'width_ratios': [ratio, 1-ratio]})
    axes[0].bar(np.arange(0, len(f_ctx)), f_ctx, label='Native firing rate')
    axes[0].bar(np.arange(0, len(f_ctx)), fstar_ctx, label='Synchronous firing rate')
    axes[0].set_title("Cortex")
    axes[0].set_xlabel("Cell index")
    axes[0].set_ylabel("Firing frequency [Hz]")
    axes[0].legend()
    axes[1].bar(np.arange(0, len(f_cbl)), f_cbl, label='Native firing rate')
    axes[1].bar(np.arange(0, len(f_cbl)), fstar_cbl, label='Synchronous firing rate')
    axes[1].set_title("Cerebellum")
    axes[1].set_xlabel("Cell index")
    axes[1].legend()
    plt.show()

def plot_ratio_native_synchronous_firing():
    t_max, f_ctx, f_cbl = get_native_firing_frequency_per_cell()
    fstar_ctx, fstar_cbl = get_synchronous_firing_frequency_per_cell(t_max)
    ratio = len(f_ctx)/(len(f_ctx) + len(f_cbl))
    f, axes = plt.subplots(1, 2, sharey='row', gridspec_kw={'width_ratios': [ratio, 1-ratio]})
    axes[0].plot(np.arange(0, len(f_ctx)), fstar_ctx/f_ctx)
    axes[0].set_title("Cortex")
    axes[0].set_xlabel("Cell index")
    axes[0].set_ylabel("Fraction of firing events inside synchronous event")
    axes[1].plot(np.arange(0, len(f_cbl)), fstar_cbl/f_cbl)
    axes[1].set_title("Cerebellum")
    axes[1].set_xlabel("Cell index")
    plt.show()

def distribution_ratios():
    t_max, f_ctx, f_cbl = get_native_firing_frequency_per_cell()
    fstar_ctx, fstar_cbl = get_synchronous_firing_frequency_per_cell(t_max)
    ratios = [i/j for i, j in zip(fstar_ctx, f_ctx)]
    ratioss = ratios.copy()
    ratioss.extend(i/j for i, j in zip(fstar_cbl, f_cbl))
    plt.hist(ratioss, label='Cerebellum')
    plt.hist(ratios, label='Cortex')
    plt.legend()
    plt.xlabel("Fraction of firing events inside synchronous event")
    plt.ylabel("Count")
    plt.show()


distribution_ratios()
