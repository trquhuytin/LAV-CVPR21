import numpy as np

from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

import matplotlib.pyplot as plt

def softmax(w, t=1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def _get_kendalls_tau(embs_list, stride, split, kt_dist, visualize=False):
    """Get nearest neighbours in embedding space and calculate Kendall's Tau."""
    num_seqs = len(embs_list)
    taus = np.zeros((num_seqs * (num_seqs - 1)))
    idx = 0
    for i in range(num_seqs):
        query_feats = embs_list[i][::stride]
        for j in range(num_seqs):
            if i == j:
                continue
            candidate_feats = embs_list[j][::stride]
            dists = cdist(query_feats, candidate_feats,
                        kt_dist)
            if visualize:
                if (i == 0 and j == 1) or split == 'val':
                    sim_matrix = []
                    for k in range(len(query_feats)):
                        sim_matrix.append(softmax(-dists[k]))
                    sim_matrix = np.array(sim_matrix, dtype=np.float32)
                    # visualize matplotlib
                    plt.imshow(sim_matrix)
                    plt.show()
            nns = np.argmin(dists, axis=1)
            taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation

            idx += 1
    # Remove NaNs.
    taus = taus[~np.isnan(taus)]
    tau = np.mean(taus)

    return tau

def evaluate_kendalls_tau(train_embs, val_embs, stride, kt_dist, visualize=False):

    train_tau = _get_kendalls_tau(train_embs, stride=stride, split='train', kt_dist=kt_dist, visualize=visualize)
    val_tau = _get_kendalls_tau(val_embs, stride=stride, split='val', kt_dist=kt_dist, visualize=visualize)

    return train_tau, val_tau