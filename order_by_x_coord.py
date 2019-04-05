import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import load_npz
from scipy.spatial.distance import euclidean
from skimage.measure import regionprops

def load_mask_to_3d_matrix(filename):
    shape_image = (188, 120)
    sparse_wrong_shape = load_npz(filename)
    wrong_shape = sparse_wrong_shape.todense()
    masks = np.zeros((shape_image[0], shape_image[1], wrong_shape.shape[1]))
    for i in range(wrong_shape.shape[1]):
        masks[:, :, i] = np.reshape(wrong_shape[:, i], shape_image)
    return masks

def determine_ordering(masks):
    x_coords = []
    for i, image in enumerate(np.rollaxis(masks, 2)):
        x_coord = regionprops((image > 0).astype(int), image)[0].centroid[0]
        x_coords.append((i, x_coord))
    x_coords_sorted = sorted(x_coords, key=lambda x: x[1])
    x_coords_sorted = [i[0] for i in x_coords_sorted]
    ordering_map = dict(zip(range(len(x_coords_sorted)), x_coords_sorted))
    return ordering_map

def order(filename):
    masks = load_mask_to_3d_matrix('/home/romano/Documents/ContinuousGlobalSynchrony/Masks/masks_{}.npz'.format(filename))
    ordering = determine_ordering(masks)
    print(len(ordering))
    fri_data = np.loadtxt('/home/romano/Documents/ContinuousGlobalSynchrony/FRIOnsetTimes/{}.csv'.format(filename), delimiter=',')
    print(fri_data.shape)
    fri_data_ordered = fri_data.copy()
    for key, val in ordering.items():
        fri_data_ordered[:, val] = fri_data[:, key]
    np.savetxt('/home/romano/Documents/ContinuousGlobalSynchrony/FRIOnsetTimesOrdered/{}.csv'.format(filename), fri_data_ordered, delimiter=',')

def order_transients(filename):
    masks = load_mask_to_3d_matrix('/home/romano/mep/ContinuousGlobalSynchrony/Masks/masks_{}.npz'.format(filename))
    ordering = determine_ordering(masks)
    transient_data = np.loadtxt('/home/romano/mep/TemporalHyperaccuity/hyperaccuity_input/{}.csv'.format(filename), delimiter=',')
    transient_data_ordered = transient_data.copy()
    for key, val in ordering.items():
        transient_data_ordered[:, val] = transient_data[:, key]
    np.savetxt('/home/romano/mep/ContinuousGlobalSynchrony/TransientsSorted/{}.csv'.format(filename), transient_data_ordered, delimiter=',')

def main():
    l = os.popen('ls /home/romano/mep/ContinuousGlobalSynchrony/Masks').read()
    l = l.split('\n')[:-1]
    l = [i[6:-4] for i in l]
    cbl = []
    ctx = []
    for i in l:
        order_transients(i)

if __name__ == '__main__':
    main()
