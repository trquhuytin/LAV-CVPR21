import os, glob
import numpy as np
import random
from natsort import natsorted
import utils

from torch.utils.data import IterableDataset

def get_steps_with_context(steps, num_context, context_stride):
    _context = np.arange(num_context-1, -1, -1)
    context_steps = np.maximum(0, steps[:, None] - _context * context_stride)
    return context_steps.reshape(-1)

def sample_frames(frames, num_context, context_stride=15):
    seq_len = len(frames)
    
    chosen_steps = np.arange(seq_len)
    steps = get_steps_with_context(chosen_steps, num_context, context_stride)

    frames = np.array(frames)[steps]
    return frames, chosen_steps


class AlignData(IterableDataset):

    def __init__(self, path, batch_size, data_config, transform=False, flatten=False):
        
        self.act_sequences = natsorted(glob.glob(os.path.join(path, '*')))
        self.n_classes = len(self.act_sequences)
        
        self.batch_size = batch_size
        self.config = data_config

        self.current_act = -1

        if transform:
            self.transform = transform
        else:
            self.transform = utils.get_totensor_transform(is_video=True)

        self.flatten = flatten

        self.action = None
        self.num_seqs = None
        self._one_vid = False

    def __len__(self):
        return self.n_classes

    def get_action_name(self, i_action):
        return os.path.basename(self.act_sequences[i_action])

    def set_action_seq(self, action, num_seqs=None):
        self.action = action
        self.num_seqs = num_seqs

    def set_spec_video(self, path):
        if not os.path.exists(path):
            raise Exception("Video doesn't exist")
        self.spec_vid = path
        self._one_vid = True

    def __iter__(self):
        
        if self.action is not None:
            if self._one_vid:
                act_sequences = [os.path.dirname(self.spec_vid)]
            else:
                act_sequences = [self.act_sequences[self.action]]
        else:
            act_sequences = self.act_sequences
        
        for _action in act_sequences:
            if self._one_vid:
                sequences = [self.spec_vid]
            else:
                sequences = natsorted(glob.glob(os.path.join(_action, '*')))
                if self.num_seqs is not None:
                    sequences = random.sample(sequences, min(self.num_seqs, len(sequences)))

            get_frame_paths = lambda x : sorted(glob.glob(os.path.join(x, '*')))

            def seq_iter():
                for seq in sequences:
                    
                    def frame_iter():
                        frames = get_frame_paths(seq)
                        num_context = self.config.NUM_CONTEXT
                        batch_step = num_context * self.batch_size

                        frames, steps = sample_frames(frames, num_context, self.config.CONTEXT_STRIDE)
                        for i in range(0, len(frames), batch_step):
                            a_frames = frames[i:i+batch_step]
                            a_x = utils.get_pil_images(a_frames)
                            a_x = self.transform(a_x)

                            if self.flatten:
                                a_x = a_x.view((a_x.shape[0], -1))
                            
                            a_name = os.path.join(os.path.basename(_action), os.path.basename(seq))

                            yield a_x, a_name, a_frames[num_context-1::num_context]

                    yield frame_iter()
            yield seq_iter()