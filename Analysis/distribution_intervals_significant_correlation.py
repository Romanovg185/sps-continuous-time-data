import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

def main():
    files = os.popen('ls IntervalsSignificantCorrelation').read().split('\n')[:-1]
    total_durations = [] 
    for my_file in files:
        tss = np.loadtxt('IntervalsSignificantCorrelation/' + my_file, delimiter=',') # [event, cell]
        durations = tss[:, 1] - tss[:, 0]
        total_durations.extend(durations)
    plt.hist(total_durations, bins=20)
    plt.show()

if __name__ == '__main__':
    main()
