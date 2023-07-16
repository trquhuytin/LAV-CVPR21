import os
import numpy as np
from imageio import imread
import cv2

import utils
import align_dataset_test as align_dataset
from config import CONFIG

from train import AlignNet

import matplotlib
matplotlib.use("Agg")

import logging
logger = logging.getLogger('matplotlib')
logger.setLevel(logging.INFO)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

import random
import itertools
import argparse

from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf

from dtw import dtw

def dist_fn(x, y):
  dist = np.sum((x-y)**2)
  return dist


def get_nn(embs, query_emb):
  dist = np.linalg.norm(embs - query_emb, axis=1)
  assert len(dist) == len(embs)
  return np.argmin(dist), np.min(dist)

def show_mat_align(D, nns):
  plt.imshow(D.T)
  plt.plot(nns, 'r', linewidth=1)
  plt.colorbar()
  plt.show()

def save_mat_align(D, nns, path):
  plt.imshow(D.T)
  plt.plot(nns, 'r', linewidth=1)
  plt.colorbar()
  plt.savefig(path)
  plt.close()

def align(query_feats, candidate_feats, use_dtw):
  """Align videos based on nearest neighbor or dynamic time warping."""
  if use_dtw:
    _, D, _, path = dtw(query_feats, candidate_feats, dist=dist_fn)
    _, uix = np.unique(path[0], return_index=True)
    nns = path[1][uix]

  else:
    nns = []
    _, D, _, _ = dtw(query_feats, candidate_feats, dist=dist_fn)
    for i in range(len(query_feats)):
      nn_frame_id, _ = get_nn(candidate_feats, query_feats[i])
      nns.append(nn_frame_id)
  return nns, D

def align_and_video(args, a_emb, b_emb, a_name, b_name, a_frames, b_frames):
    nns_a, dist_mat_a = align(a_emb, a_emb, use_dtw=args.use_dtw)
    save_mat_align(dist_mat_a, nns_a, args.dest+'Self-{}-align-{}-stride-{}-dtw-{}-bs-{}.png'.format(a_name, args.mode,
                                                                        args.stride, args.use_dtw, args.batch_size).replace('/', '_'))

    nns_b, dist_mat_b = align(b_emb, b_emb, use_dtw=args.use_dtw)
    save_mat_align(dist_mat_b, nns_b, args.dest+'Self-{}-align-{}-stride-{}-dtw-{}-bs-{}.png'.format(b_name, args.mode,
                                                                        args.stride, args.use_dtw, args.batch_size).replace('/', '_'))

    nns, dist_mat = align(a_emb[::args.stride], b_emb[::args.stride], use_dtw=args.use_dtw)

    print(dist_mat.shape)

    save_mat_align(dist_mat, nns, args.dest+'{}-{}-align-{}-stride-{}-dtw-{}-bs-{}.png'.format(a_name, b_name, args.mode, 
                                                                        args.stride, args.use_dtw, args.batch_size).replace('/', '_'))

    aligned_imgs = []
    a_frames = a_frames[::args.stride]
    b_frames = b_frames[::args.stride]

    max_len = max(len(a_frames), len(b_frames))

    for i in range(max_len):
        
        aimg = imread(a_frames[min(i, len(a_frames)-1)])
        aimg = cv2.resize(aimg, (224, 224))
        bimg_nn = imread(b_frames[nns[min(i, len(nns)-1)]])
        bimg_nn = cv2.resize(bimg_nn, (224, 224))

        bimg_i = imread(b_frames[min(i, len(b_frames)-1)])
        bimg_i = cv2.resize(bimg_i, (224, 224))

        print('Aligned  {} - {}'.format(min(i, len(a_frames)-1), nns[min(i, len(a_frames)-1)]))

        ab_img_nn = np.concatenate((aimg, bimg_nn), axis=1)
        ab_img_i = np.concatenate((aimg, bimg_i), axis=1)

        ab_img = np.concatenate((ab_img_nn, ab_img_i), axis=0)
        aligned_imgs.append(ab_img)
    
    def make_video(img):

        frames = [] # for storing the generated images
        fig = plt.figure()

        print('LEN: ', len(img))

        for i in range(len(img)):
            frames.append([plt.imshow(img[i],animated=True)])

        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                        repeat_delay=1000)
        ani.save(args.dest+'{}-{}-align-{}-stride-{}-dtw-{}-bs-{}.mp4'.format(a_name, b_name, args.mode, 
                                                                            args.stride, args.use_dtw, args.batch_size).replace('/', '_'))
        plt.close(fig)

    make_video(aligned_imgs)

def main(args):

    model = AlignNet.load_from_checkpoint(args.model_path, map_location=args.device)
    model.to(args.device)

    if args.mode == 'train':
        model.train()
    else:
        model.eval()

    eval_transforms = utils.get_transforms(augment=False)

    random.seed(args.seed)
    data = align_dataset.AlignData(args.data_path, args.batch_size, CONFIG.DATA, transform=eval_transforms, flatten=False)

    for i in range(data.n_classes):
        # get 2 videos of 0th action
        data.set_action_seq(action=i, num_seqs=args.num_seqs)

        embeddings = []
        frame_paths = []
        names = []

        for act_iter in iter(data):
            for seq_iter in act_iter:

                seq_embs = []
                seq_fpaths = []
                for _, batch in enumerate(seq_iter):
                    
                    a_X, a_name, a_frames = batch
                    
                    print(a_X.shape)
                    print(a_name)
                    
                    a_emb = model(a_X.to(args.device).unsqueeze(0))
                    print(a_emb.shape)

                    seq_embs.append(a_emb.squeeze(0).detach().cpu().numpy())
                    seq_fpaths.extend(a_frames)
                
                seq_embs = np.concatenate(seq_embs, axis=0)
                embeddings.append(seq_embs)
                frame_paths.append(seq_fpaths)
                names.append(a_name)

        print(len(embeddings))
        print(len(frame_paths))

        print(embeddings[0].shape)
        print(embeddings[1].shape)
        print(frame_paths[0][-1])
        print(frame_paths[1][-1])
        print(names)
        
        for i, j in itertools.combinations(range(len(embeddings)), 2):
            align_and_video(args, embeddings[i], embeddings[j], names[i], names[j], frame_paths[i], frame_paths[j])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--dest', type=str, default='./')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--use_dtw', dest='use_dtw', action='store_true')

    parser.add_argument('--num_seqs', type=int, default=2)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    main(args)
