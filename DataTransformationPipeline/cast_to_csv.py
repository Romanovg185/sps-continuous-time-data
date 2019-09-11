import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse import issparse
import os

def cast_single_to_csv(file_path, recording_rate_hz):
    x = loadmat('FRIOnsetTimes/' + file_path)['results'][0,0]
    l = []
    for f in x:
        if issparse(f):
            l.append(f)
    m = min([l[i].shape[0] for i in range(2)])
    q = [i for i in l if i.shape[0] == m][0]
    cells, times = q.nonzero()
    times = times / recording_rate_hz
    tot = [[] for i in range(max(cells) + 1)]
    for cell, time in zip(cells, times):
        tot[cell - 1].append(time)
    m = np.full((len(tot), max([len(i) for i in tot])), np.nan)
    for i, time in enumerate(tot):
        m[i, :len(time)] = time
    m = m.T
    np.savetxt('FRIOnsetTimes/' + file_path[:-3] + 'csv', m, delimiter=',')

def cast_to_csv(recording_rate_hz):
    for i in os.listdir('FRIOnsetTimes'):
        if i[-3:] == 'mat':
            cast_single_to_csv(i, recording_rate_hz)

    
