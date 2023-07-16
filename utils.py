import os, glob
import io
import numpy as np
from imageio import imread
from PIL import Image

import torch
from torchvision import transforms

import matplotlib.pyplot as plt

import preprocessing.transforms_video as tv
from config import CONFIG

import pytorch_lightning as pl

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)

def normalize(img):
    return img / 255.

def get_totensor_transform(is_video):
    if is_video:
        return transforms.Compose([tv.ToTensorVideo()])
    else:
        return transforms.Compose([transforms.ToTensor()])

def get_images(paths):

    imgs = []
    for p in paths:
        imgs.append(normalize(imread(p)))

    return np.array(imgs, dtype=np.float32)

def get_transformed_images(paths, transform):

    imgs = []
    for p in paths:
        imgs.append(transform(Image.open(p)))

    imgs = torch.stack(imgs)
    
    return imgs

def get_pil_images(paths):

    imgs = []
    for p in paths:
        imgs.append(Image.open(p))
    
    return imgs

def get_transforms(augment):
    
    seq_transforms = []

    if augment:
        seq_transforms.append(tv.ColorJitterVideo(
            CONFIG.AUGMENTATION.BRIGHTNESS_DELTA,
            CONFIG.AUGMENTATION.CONTRAST_DELTA,
            CONFIG.AUGMENTATION.HUE_DELTA,
            CONFIG.AUGMENTATION.SATURATION_DELTA
        ))

        if not CONFIG.AUGMENTATION.RANDOM_CROP:
            seq_transforms.append(tv.ResizeVideo(size=(CONFIG.DATA.IMAGE_SIZE, CONFIG.DATA.IMAGE_SIZE)))

        seq_transforms.append(tv.ToTensorVideo())

        if CONFIG.AUGMENTATION.RANDOM_FLIP:
            seq_transforms.append(tv.RandomHorizontalFlipVideo(p=0.5))
        
        if CONFIG.AUGMENTATION.RANDOM_CROP:
            seq_transforms.append(tv.RandomResizedCropVideo(size=CONFIG.DATA.IMAGE_SIZE))
    
    else:
        if CONFIG.AUGMENTATION.RANDOM_CROP:
            seq_transforms.append(tv.ToTensorVideo())
            seq_transforms.append(tv.CenterResizedCropVideo(size=CONFIG.DATA.IMAGE_SIZE))
        else:
            seq_transforms.append(tv.ResizeVideo(size=(CONFIG.DATA.IMAGE_SIZE, CONFIG.DATA.IMAGE_SIZE)))
            seq_transforms.append(tv.ToTensorVideo())

    seq_transforms.append(tv.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]))

    return transforms.Compose(seq_transforms)

def arg_to_numpy(f):

    def wrapper(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return f(x)
    return wrapper

@arg_to_numpy
def plot_to_image(arr):
    
    arr = arr.squeeze()
    figure = plt.figure()
    plt.imshow(arr)
    plt.colorbar()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    plt.close(figure)
    buf.seek(0)
    img = imread(buf)

    return img.transpose((2, 0, 1))

def _make_trainable(module):
    """Unfreeze a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module, train_bn=True):
    """Freeze the layers of a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
        Otherwise, they will be set to eval mode along with the other modules.
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)

def freeze(module, n=-1, train_bn=True):
    """Freeze the layers up to index n.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    n : int
        By default, all the layers will be frozen. Otherwise, an integer
        between 0 and `len(module.children())` must be given.
    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
    """
    idx = 0
    children = list(module.children())
    n_max = len(children) if n == -1 else int(n)
    for child in children:
        if idx < n_max:
            #print('-----------------',child,'-------------')
            _recursive_freeze(module=child, train_bn=train_bn)
        else:
            _make_trainable(module=child)

def _recursive_freeze_bn_only(module):
    """Freeze the BN-layers of a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    """
    children = list(module.children())
    if not children:
        if isinstance(module, BN_TYPES):
            print('Froze ',module)
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the other layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze_bn_only(module=child)

def freeze_bn_only(module, n=-1):
    """Freeze the BN-layers up to index n.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    n : int
        By default, all the BN-layers will be frozen. Otherwise, an integer
        between 0 and `len(module.children())` must be given.
    """
    idx = 0
    children = list(module.children())
    n_max = len(children) if n == -1 else int(n)
    for child in children:
        if idx < n_max:
            #print('-----------------',child,'-------------')
            _recursive_freeze_bn_only(module=child)
        else:
            _make_trainable(module=child)

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        filepath,
        prefix="model",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.filepath = filepath
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

        os.makedirs(self.filepath, exist_ok=True)

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}lAV_epoch={epoch}_step={global_step}.ckpt"
            ckpt_path = os.path.join(self.filepath, filename)
            trainer.save_checkpoint(ckpt_path)
import os, glob
import io
import numpy as np
from imageio import imread
from PIL import Image

import torch
from torchvision import transforms

import matplotlib.pyplot as plt

import preprocessing.transforms_video as tv
from config import CONFIG

import pytorch_lightning as pl

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)

def normalize(img):
    return img / 255.

def get_totensor_transform(is_video):
    if is_video:
        return transforms.Compose([tv.ToTensorVideo()])
    else:
        return transforms.Compose([transforms.ToTensor()])

def get_images(paths):

    imgs = []
    for p in paths:
        imgs.append(normalize(imread(p)))

    return np.array(imgs, dtype=np.float32)

def get_transformed_images(paths, transform):

    imgs = []
    for p in paths:
        imgs.append(transform(Image.open(p)))

    imgs = torch.stack(imgs)
    
    return imgs

def get_pil_images(paths):

    imgs = []
    for p in paths:
        imgs.append(Image.open(p))
    
    return imgs

def get_transforms(augment):
    
    seq_transforms = []

    if augment:
        seq_transforms.append(tv.ColorJitterVideo(
            CONFIG.AUGMENTATION.BRIGHTNESS_DELTA,
            CONFIG.AUGMENTATION.CONTRAST_DELTA,
            CONFIG.AUGMENTATION.HUE_DELTA,
            CONFIG.AUGMENTATION.SATURATION_DELTA
        ))

        if not CONFIG.AUGMENTATION.RANDOM_CROP:
            seq_transforms.append(tv.ResizeVideo(size=(CONFIG.DATA.IMAGE_SIZE, CONFIG.DATA.IMAGE_SIZE)))

        seq_transforms.append(tv.ToTensorVideo())

        if CONFIG.AUGMENTATION.RANDOM_FLIP:
            seq_transforms.append(tv.RandomHorizontalFlipVideo(p=0.5))
        
        if CONFIG.AUGMENTATION.RANDOM_CROP:
            seq_transforms.append(tv.RandomResizedCropVideo(size=CONFIG.DATA.IMAGE_SIZE))
    
    else:
        if CONFIG.AUGMENTATION.RANDOM_CROP:
            seq_transforms.append(tv.ToTensorVideo())
            seq_transforms.append(tv.CenterResizedCropVideo(size=CONFIG.DATA.IMAGE_SIZE))
        else:
            seq_transforms.append(tv.ResizeVideo(size=(CONFIG.DATA.IMAGE_SIZE, CONFIG.DATA.IMAGE_SIZE)))
            seq_transforms.append(tv.ToTensorVideo())

    seq_transforms.append(tv.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]))

    return transforms.Compose(seq_transforms)

def arg_to_numpy(f):

    def wrapper(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return f(x)
    return wrapper

@arg_to_numpy
def plot_to_image(arr):
    
    arr = arr.squeeze()
    figure = plt.figure()
    plt.imshow(arr)
    plt.colorbar()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    plt.close(figure)
    buf.seek(0)
    img = imread(buf)

    return img.transpose((2, 0, 1))

def _make_trainable(module):
    """Unfreeze a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module, train_bn=True):
    """Freeze the layers of a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
        Otherwise, they will be set to eval mode along with the other modules.
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)

def freeze(module, n=-1, train_bn=True):
    """Freeze the layers up to index n.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    n : int
        By default, all the layers will be frozen. Otherwise, an integer
        between 0 and `len(module.children())` must be given.
    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
    """
    idx = 0
    children = list(module.children())
    n_max = len(children) if n == -1 else int(n)
    for child in children:
        if idx < n_max:
            #print('-----------------',child,'-------------')
            _recursive_freeze(module=child, train_bn=train_bn)
        else:
            _make_trainable(module=child)

def _recursive_freeze_bn_only(module):
    """Freeze the BN-layers of a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    """
    children = list(module.children())
    if not children:
        if isinstance(module, BN_TYPES):
            print('Froze ',module)
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the other layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze_bn_only(module=child)

def freeze_bn_only(module, n=-1):
    """Freeze the BN-layers up to index n.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    n : int
        By default, all the BN-layers will be frozen. Otherwise, an integer
        between 0 and `len(module.children())` must be given.
    """
    idx = 0
    children = list(module.children())
    n_max = len(children) if n == -1 else int(n)
    for child in children:
        if idx < n_max:
            #print('-----------------',child,'-------------')
            _recursive_freeze_bn_only(module=child)
        else:
            _make_trainable(module=child)

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        filepath,
        prefix="model",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.filepath = filepath
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

        os.makedirs(self.filepath, exist_ok=True)

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}lAV_epoch={epoch}_step={global_step}.ckpt"
            ckpt_path = os.path.join(self.filepath, filename)
            trainer.save_checkpoint(ckpt_path)
