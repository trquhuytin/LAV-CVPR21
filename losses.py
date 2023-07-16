import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from soft_dtw import SoftDTW

def calc_distance_matrix(x, y):
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    dist = torch.pow(x - y, 2).sum(3)
    return dist


class Contrastive_IDM(nn.Module):

    def __init__(self, sigma, margin, debug=False):
        super(Contrastive_IDM, self).__init__()

        self.sigma = sigma
        self.margin = margin
        self.debug = debug

    def forward(self, dist, idx, seq_len, logger=None):

        grid_x, grid_y = torch.meshgrid(idx, idx)

        prob = F.relu(self.margin - dist)

        weights_orig = 1 + torch.pow(grid_x - grid_y, 2)

        diff = torch.abs(grid_x - grid_y) - (self.sigma / seq_len)
        
        _ones = torch.ones_like(diff)
        _zeros = torch.zeros_like(diff)
        weights_neg = torch.where(diff > 0, weights_orig, _zeros)

        weights_pos = torch.where(diff > 0, _zeros, _ones)

        if not self.training and self.debug and logger:
            logger.experiment.add_image('idm_diff', utils.plot_to_image(diff), 0, dataformats='CHW')
            logger.experiment.add_image('idm_weights_pos', utils.plot_to_image(weights_pos), 0, dataformats='CHW')
            logger.experiment.add_image('idm_weights_neg', utils.plot_to_image(weights_neg), 0, dataformats='CHW')
            logger.experiment.add_image('idm_prob', utils.plot_to_image(prob), 0, dataformats='CHW')
        
        idm = weights_neg * prob + weights_pos * dist

        return torch.sum(idm), idm

class LAV(nn.Module):

    def __init__(self, alpha, sigma, margin, num_frames, dtw_gamma, dtw_normalize, debug=False):
        super(LAV, self).__init__()

        self.alpha = alpha
        self.debug = debug
        self.N = num_frames

        self.dtw_loss = SoftDTW(gamma=dtw_gamma, normalize=dtw_normalize)

        self.inverse_idm = Contrastive_IDM(sigma=sigma, margin=margin, debug=debug)

    def forward(self, a_emb, b_emb, a_idx, b_idx, a_len, b_len, logger=None):

        pos_loss = self.dtw_loss(a_emb, b_emb)

        # frame level loss
        dist_a = calc_distance_matrix(a_emb, a_emb).squeeze(0)
        dist_b = calc_distance_matrix(b_emb, b_emb).squeeze(0)

        idm_a, _ = self.inverse_idm(dist_a, a_idx, a_len, logger=logger)
        idm_b, _ = self.inverse_idm(dist_b, b_idx, b_len, logger=logger)

        total_loss = pos_loss + self.alpha * (idm_a + idm_b)
        total_loss = total_loss / self.N

        if not self.training and self.debug and logger:
            logger.experiment.add_image('dist_a', utils.plot_to_image(dist_a), 0, dataformats='CHW')
            logger.experiment.add_image('dist_b', utils.plot_to_image(dist_b), 0, dataformats='CHW')

        return total_loss

class TCC(nn.Module):

    def __init__(self, channels, temperature, var_lambda, debug=False):
        super(TCC, self).__init__()

        self.debug = debug
        self.channels = channels
        self.temperature = temperature
        self.var_lambda = var_lambda

    def _pairwise_distance(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        dist = torch.pow(x - y, 2).sum(2)
        return dist

    def _get_scaled_similarity(self, emb_a, emb_b):

        sim = -1. * self._pairwise_distance(emb_a, emb_b)

        scaled_similarity = sim / self.channels
        scaled_similarity = scaled_similarity / self.temperature

        return scaled_similarity

    def _tcc_loss(self, emb_a, emb_b, idxes):

        sim_ab = self._get_scaled_similarity(emb_a, emb_b)
        softmaxed_sim_ab = F.softmax(sim_ab, dim=-1)

        soft_nn = torch.matmul(softmaxed_sim_ab, emb_b)

        sim_ba = self._get_scaled_similarity(soft_nn, emb_a)

        labels = idxes

        beta = F.softmax(sim_ba, dim=-1)
        preds = torch.sum(beta * idxes, dim=-1, keepdim=True)

        pred_var = torch.sum(torch.pow(idxes - preds, 2.) * beta, axis=1)
        pred_var_log = torch.log(pred_var)

        squared_error = torch.pow(labels.squeeze() - preds.squeeze(), 2.)
        return torch.sum(torch.exp(-pred_var_log) * squared_error + self.var_lambda * pred_var_log)


    def forward(self, emb_a, emb_b, idx_a, idx_b, logger=None):
        
        emb_a = emb_a.squeeze(0)
        emb_b = emb_b.squeeze(0)

        loss_ab = self._tcc_loss(emb_a, emb_b, idx_a)
        loss_ba = self._tcc_loss(emb_b, emb_a, idx_b)

        return (loss_ab + loss_ba) / (emb_a.size(0) + emb_b.size(0))


class TCN(nn.Module):

    def __init__(self, reg_lambda=0.002):
        super(TCN, self).__init__()

        self.reg_lambda = reg_lambda

    def _npairs_loss(self, labels, embeddings_anchor, embeddings_positive):
        """Returns n-pairs metric loss."""
        square = lambda x : torch.pow(x, 2)
        reg_anchor = torch.mean(torch.sum(square(embeddings_anchor), 1))
        reg_positive = torch.mean(torch.sum(
            square(embeddings_positive), 1))
        l2loss = 0.25 * self.reg_lambda * (reg_anchor + reg_positive)

        # Get per pair similarities.
        similarity_matrix = torch.matmul(
            embeddings_anchor, embeddings_positive.t())

        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        lshape = labels.shape

        # Add the softmax loss.
        xent_loss = F.cross_entropy(
            input=similarity_matrix, target=labels)
        #xent_loss = tf.reduce_mean(xent_loss)

        return l2loss + xent_loss


    def single_sequence_loss(self, embs, num_steps):
        """Returns n-pairs loss for a single sequence."""

        labels = torch.arange(num_steps)
        embeddings_anchor = embs[0::2]
        embeddings_positive = embs[1::2]
        loss = self._npairs_loss(labels, embeddings_anchor, embeddings_positive)
        return loss

    def forward(self, embs, num_steps):
        return self.single_sequence_loss(embs.squeeze(0), num_steps)