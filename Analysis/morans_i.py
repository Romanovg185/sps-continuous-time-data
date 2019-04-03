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

# Different distance matrices
def overlap_based_distance(masks):
    weight_matrix = np.zeros((masks.shape[2], masks.shape[2]))
    for i, image_0 in enumerate(np.rollaxis(masks, 2)):
        #print('{}/{}'.format(i+1, masks.shape[2]))
        for j, image_1 in enumerate(np.rollaxis(masks, 2)):
            overlap = np.logical_and(image_0, image_1)
            if overlap.sum() > 0:
                weight_matrix[i, j] = 1
    return weight_matrix

def centroid_inverse_distance(masks):
    weight_matrix = np.zeros((masks.shape[2], masks.shape[2]))
    for i, image_0 in enumerate(np.rollaxis(masks, 2)):
        #print('{}/{}'.format(i+1, masks.shape[2]))
        for j, image_1 in enumerate(np.rollaxis(masks, 2)):
            centroid_0 = regionprops((image_0 > 0).astype(int), image_0)[0].centroid
            centroid_1 = regionprops((image_1 > 0).astype(int), image_1)[0].centroid
            dist = euclidean(centroid_0, centroid_1)
            if dist > 0:
                weight_matrix[i, j] = 1/dist
    return weight_matrix

def moran_i(filename):
    masks = load_mask_to_3d_matrix('/home/romano/mep/ContinuousGlobalSynchony/Masks/masks_{}.npz'.format(filename))
    weight_matrix = centroid_inverse_distance(masks)

    x = np.loadtxt('/home/romano/mep/ContinuousGlobalSynchony/SynchronousEventParticipatingNeurons/{}_participating_neurons_{}.csv'.format(filename[:3], filename[4:]).replace('_results_', '_'), delimiter=',')
    x = np.sum(x, axis=1)
    xbar = np.mean(x)
    morans_i = 0
    for i, eli in enumerate(x):
        for j, elj in enumerate(x):
            morans_i += weight_matrix[i, j]*(eli - xbar)*(elj - xbar)
    denum = 0
    for i in x:
        denum += (i - xbar)**2
    morans_i /= denum
    morans_i *= len(x)
    morans_i /= np.sum(weight_matrix)
    return morans_i

def main():
    l = os.popen('ls /home/romano/mep/ContinuousGlobalSynchony/Masks').read()
    l = l.split('\n')[:-1]
    l = [i[6:-4] for i in l]
    cbl = []
    ctx = []
    for i in l:
        print(i)
        moran = moran_i(i)
        if i[:3] == 'cbl':
            cbl.append(moran)
        if i[:3] == 'ctx':
            ctx.append(moran)
    plt.subplot(1, 2, 1)
    plt.hist(cbl)
    plt.ylabel('Count')
    plt.xlabel("Moran's I")
    plt.title('Cerebellum')
    plt.subplot(1, 2, 2)
    plt.hist(ctx)
    plt.xlabel("Moran's I")
    plt.title('Cortex')
    plt.show()

if __name__ == '__main__':
    main()
