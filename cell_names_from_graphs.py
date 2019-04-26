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
for my_file in files:
    if 'same' in my_file or 'cross' in my_file:
        continue
    l = get_edges(loc + my_file)
    pairs_of_names = [get_names_from_edge(loc + my_file, *i) for i in l]
    cbl_cbl = set()
    ctx_ctx = set()
    cross_cbl = set()
    cross_ctx = set()
    for first, sec in pairs_of_names:
        if first[:3] == 'Cbl' and sec[:3] == 'Cbl':
            cbl_cbl.add(int(first[3:]))
            cbl_cbl.add(int(sec[3:]))
        if first[:3] == 'Cbl' and sec[:3] == 'Ctx':
            cross_cbl.add(int(first[3:]))
            cross_ctx.add(int(sec[3:]))
        if first[:3] == 'Ctx' and sec[:3] == 'Cbl':
            cross_ctx.add(int(first[3:]))
            cross_cbl.add(int(sec[3:]))
        if first[:3] == 'Ctx' and sec[:3] == 'Ctx':
            ctx_ctx.add(int(first[3:]))
            ctx_ctx.add(int(sec[3:]))
    print(my_file)
    print(cross_cbl)
    print(cross_ctx)
    cross_ctx = list(cross_ctx)
    cross_cbl = list(cross_cbl)
    while len(cross_ctx) < len(cross_cbl):
        cross_ctx.append(999999)
    while len(cross_cbl) < len(cross_ctx):
        cross_cbl.append(999999)
    with open(target + my_file[:-4] + '_cblcbl.csv', 'w') as f:
        for i in sorted(list(cbl_cbl)):
            f.write(f'{i}\n')
    with open(target + my_file[:-4] + '_ctxctx.csv', 'w') as f:
        for i in sorted(list(ctx_ctx)):
            f.write(f'{i}\n')
    with open(target + my_file[:-4] + '_cross.csv', 'w') as f:
        for i, j in zip(sorted(cross_cbl), sorted(cross_ctx)):
            istar = i if i != 999999 else ' '
            jstar = j if j != 999999 else ' '
            f.write(f'{istar},{jstar}\n')
        
