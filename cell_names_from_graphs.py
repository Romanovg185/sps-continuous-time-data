import numpy as np
import matplotlib.pyplot as plt
import os

def get_edges(filepath):
    with open(filepath) as f:
        indices_edge = {i for i, el in enumerate(f) if 'edge' in el}
    with open(filepath) as f:
        froms = [el for i, el in enumerate(f) if i-1 in indices_edge]
    with open(filepath) as f:
        tos = [el for i, el in enumerate(f) if i-2 in indices_edge]
    froms = [list(i) for i in froms]
    froms = [int(''.join([j for j in i if j.isnumeric()])) for i in froms]
    tos = [list(i) for i in tos]
    tos = [int(''.join([j for j in i if j.isnumeric()])) for i in tos]
    indices = list(zip(froms, tos))
    return indices

def get_names_from_edge(filepath, source, target):
    with open(filepath) as f:
        ids = [el for el in f if 'id' in el]
    with open(filepath) as f:
        labs = [el for el in f if 'label' in el]
    ids = [list(i) for i in ids]
    ids = [int(''.join([j for j in i if j.isnumeric()])) for i in ids]
    labs = [i[11:-2] for i in labs]
    z = zip(ids, labs)
    source_name = [j for i, j in z if i == source][0]
    z = zip(ids, labs)
    target_name = [j for i, j in z if i == target][0]
    return source_name, target_name

loc = '/home/romano/Documents/ContinuousGlobalSynchrony/Graphs/'
target = '/home/romano/Documents/ContinuousGlobalSynchrony/Indices/'
files = os.popen(f'ls {loc}').read().split('\n')[:-1]
print(files)
for my_file in files:
    l = get_edges(loc + my_file)
    pairs_of_names = [get_names_from_edge(loc + my_file, *i) for i in l]
    with open(target + my_file[:-3] + 'csv', 'w') as f:
        for pair in pairs_of_names:
            f.write(f'{pair[0]},{pair[1]}\n')

        
