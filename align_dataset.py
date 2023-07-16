import os, glob
import numpy as np
import random
import utils

from torch.utils.data import Dataset

def get_steps_with_context(steps, num_context, context_stride):
    _context = np.arange(num_context-1, -1, -1)
    context_steps = np.maximum(0, steps[:, None] - _context * context_stride)
    return context_steps.reshape(-1)

def sample_frames(frames, num_frames, num_context, frame_stride, 
                        sampling='offset_uniform', random_offset=1, context_stride=15, is_tcn=False, tcn_window=5):
    
    seq_len = len(frames)

    if sampling == 'stride':

        offset = random.randint(0, max(1, seq_len - frame_stride * num_frames)-1)
        steps = np.arange(offset, offset + frame_stride * num_frames + 1, frame_stride)
        # cap at max length
        steps = np.minimum(steps, seq_len-1)
        steps = steps[:num_frames]

    elif sampling == 'offset_uniform':
        
        def _sample_random(offset):
            assert offset <= seq_len, "Offset is greater than the Sequence length"
            steps = np.arange(offset, seq_len)
            random.shuffle(steps)
            steps = steps[:num_frames]
            steps = np.sort(steps)
            return steps
        
        def _sample_all():
            return np.arange(0, num_frames)

        if num_frames < seq_len - random_offset:
            steps = _sample_random(random_offset)
        else:
            steps = _sample_all()
    elif sampling == 'segment_uniform':
        
        if num_frames > seq_len:
            steps = np.arange(num_frames)
        else:
            r = num_frames - seq_len % num_frames

            if r < num_frames:
                steps = np.concatenate([np.arange(seq_len), np.arange(r)])
            else:
                steps = np.arange(seq_len)
            f = len(steps) / num_frames

            sampled_idxes = np.arange(num_frames) * f + np.array(random.choices(range(np.int(f)), k=num_frames))
            sampled_idxes = sampled_idxes.astype(np.int32)

            steps = np.sort(steps[sampled_idxes])

    elif sampling == 'all':
        steps = np.arange(0, seq_len)
    else:
        raise Exception("{} not implemented.".format(sampling))

    if is_tcn:
        pos_steps = steps - np.array(random.choices(range(1, tcn_window + 1), k=len(num_frames)))
        steps = np.stack([pos_steps, steps])
        steps = steps.T.reshape((-1, ))
    
    steps = np.minimum(steps, seq_len-1)
    chosen_steps = steps.astype(np.float32) / seq_len
    steps = get_steps_with_context(steps, num_context, context_stride)

    frames = np.array(frames)[steps]

    return frames, chosen_steps, float(seq_len)


class AlignData(Dataset):

    def __init__(self, path, num_frames, data_config, neg_example=False, transform=False, flatten=False):

        self.act_sequences = sorted(glob.glob(os.path.join(path, '*')))
        self.n_sequences = len(self.act_sequences)
        
        self.n_classes = len(self.act_sequences)
        self.num_frames = num_frames
        self.config = data_config

        self.neg_example = neg_example

        if transform:
            self.transform = transform
        else:
            self.transform = utils.get_totensor_transform(is_video=True)

        self.flatten = flatten

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):

        a = self.act_sequences[idx]
        
        b = a
        while a == b:
            b = random.choice(self.act_sequences)

        assert a != b, "Same sequences sampled!"

        config = self.config
        get_frame_paths = lambda x : sorted(glob.glob(os.path.join(x, '*')))
        
        a_frames = get_frame_paths(a)
        # a_frames, a_chosen_steps, a_seq_len = sample_frames(a_frames, num_frames=self.num_frames, num_context=config.NUM_CONTEXT, 
        #                                             frame_stride=config.FRAME_STRIDE, sampling=config.SAMPLING_STRATEGY, 
        #                                             random_offset=config.RANDOM_OFFSET, context_stride=config.CONTEXT_STRIDE,
        #                                             is_tcn=config.TCN.IS_TCN, tcn_window=config.TCN.POS_WINDOW)
    
        a_frames, a_chosen_steps, a_seq_len = sample_frames(a_frames, num_frames=self.num_frames, num_context=config.NUM_CONTEXT, 
                                                frame_stride=config.FRAME_STRIDE, sampling=config.SAMPLING_STRATEGY, 
                                                random_offset=config.RANDOM_OFFSET, context_stride=config.CONTEXT_STRIDE)

        b_frames = get_frame_paths(b)
        # b_frames, b_chosen_steps, b_seq_len = sample_frames(b_frames, num_frames=self.num_frames, num_context=config.NUM_CONTEXT, 
        #                                             frame_stride=config.FRAME_STRIDE, sampling=config.SAMPLING_STRATEGY, 
        #                                             random_offset=config.RANDOM_OFFSET, context_stride=config.CONTEXT_STRIDE,
        #                                             is_tcn=config.TCN.IS_TCN, tcn_window=config.TCN.POS_WINDOW)

        b_frames, b_chosen_steps, b_seq_len = sample_frames(b_frames, num_frames=self.num_frames, num_context=config.NUM_CONTEXT, 
                                            frame_stride=config.FRAME_STRIDE, sampling=config.SAMPLING_STRATEGY, 
                                            random_offset=config.RANDOM_OFFSET, context_stride=config.CONTEXT_STRIDE)

        a_x = utils.get_pil_images(a_frames)
        b_x = utils.get_pil_images(b_frames)

        a_x = self.transform(a_x)
        b_x = self.transform(b_x)

        a_name = 'Vid_{}'.format(os.path.basename(a))
        b_name = 'Vid_{}'.format(os.path.basename(b))

        result = [[a_x, a_name, a_chosen_steps, a_seq_len], [b_x, b_name, b_chosen_steps, b_seq_len]]

        if self.flatten:
            for item in result:
                item[0] = item[0].view((item[0].shape[0], -1))
        
        return result