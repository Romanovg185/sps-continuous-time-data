import sys
sys.path.append("..")
from convolve_simplified import convolve_with_kernel_two_sigma
import numpy as np
import matplotlib.pyplot as plt

"""
Plots a histogram of the widths in seconds of all synchronous events detected using the 2 std of raw data DF/F threshold
:param m_raw: Raw inferred onset time matrix
"""
def plot_histogram_synchronous_event_width_2_sigma(m_raw):
    convolution_output = convolve_with_kernel_two_sigma(m_raw)
    widths = [i[0] for i in convolution_output]
    plt.hist(widths)
    plt.xlabel('Time [sec]')
    plt.ylabel('Count')
    plt.title(r'Duration of synchronous event using the $2\sigma$ threshold')
    plt.show()

if __name__ == '__main__':
    m_raw = np.loadtxt('/home/romano/mep/ContinuousGlobalSynchrony/FRIOnsetTimes/cbl_results_1_4000_fr_16082018_161839_dt.csv', delimiter=',')
    plot_histogram_synchronous_event_width_2_sigma(m_raw)
