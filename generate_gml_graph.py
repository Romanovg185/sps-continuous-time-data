import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx

def main():
    l = os.popen("ls FourBoxRawData").read().split('\n')
    l = ["/home/romano/mep/ContinuousGlobalSynchrony/FourBoxRawData/" + i for i in l[:-1]]
    fill_color_cbl = "#872a0e"
    fill_color_ctx = "#634289"
    threshold = 5
    for path in l:
        m = np.loadtxt(path, delimiter=',')
        n_tot = m.shape[0] - 1
        n_cbl = np.where(np.isnan(m[0, :]))[0][0]
        m_cblcbl = m[:n_cbl, :n_cbl]
        m_cblctx = m[n_cbl+1:, :n_cbl]
        m_ctxctx = m[n_cbl+1:, n_cbl+1:]
        filename = path[73:-4] + ".gml"
        # Fixing matrices
        m = np.zeros((m.shape[0] - 1, m.shape[1] - 1))
        m[:n_cbl, :n_cbl] = m_cblcbl
        m[n_cbl:, :n_cbl] = m_cblctx
        m[n_cbl:, n_cbl:] = m_ctxctx
        m = np.tril(m)
        mask = m >= threshold
        indices = np.argwhere(mask)

        indices_used = set()
        for i in indices:
            indices_used.add(i[0])
            indices_used.add(i[1])
        indices_used = sorted(list(indices_used))
        names = ['Cbl{}'.format(i+1) for i in indices_used if i < n_cbl]
        names.extend('Ctx{}'.format(i+1 - n_cbl) for i in indices_used if i >= n_cbl)
        groups = [1 for i in indices_used if i < n_cbl]
        groups.extend(2 for i in indices_used if i >= n_cbl)
        colors = [fill_color_cbl for i in indices_used if i < n_cbl]
        colors.extend(fill_color_ctx for i in indices_used if i >= n_cbl)
        borders = ['#aba7c4' for _ in indices_used]

        my_graph = nx.Graph()
        for i in names:
            my_graph.add_node(i)
        group_dict = dict(zip(names, groups))
        fill_dict = dict(zip(names, colors))
        border_dict = dict(zip(names, borders))
        nx.set_node_attributes(my_graph, group_dict, "group")
        nx.set_node_attributes(my_graph, fill_dict, "fill")
        nx.set_node_attributes(my_graph, border_dict, "border")

        index_named = []
        weights = []
        for i in indices:
            if i[0] < n_cbl:
                first = 'Cbl{}'.format(i[0]+1)
            else:
                first = 'Ctx{}'.format(i[0]+1 - n_cbl)
            if i[1] < n_cbl:
                second = 'Cbl{}'.format(i[1]+1)
            else:
                second = 'Ctx{}'.format(i[1]+1 - n_cbl)
            index_named.append((first, second))
            if first[:3] == second[:3]:
                weights.append(int(m[i[0], i[1]]))
            else:
                weights.append(0)
        for index in index_named:
            my_graph.add_edge(index[0], index[1])
        weight_dict = dict(zip(index_named, weights))
        nx.set_edge_attributes(my_graph, weight_dict, "value")
        print(filename)
        nx.write_gml(my_graph, '/home/romano/mep/ContinuousGlobalSynchrony/Graphs/{}_same.gml'.format(filename[:-4]))


main()
