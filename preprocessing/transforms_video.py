import numbers
import random
from PIL import Image
import collections

import torch
from torchvision.transforms import (
    RandomCrop,
    RandomResizedCrop,
    Resize,
    ToTensor,
    ColorJitter,
)

from . import functional_video as F

import sys
if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = [
    "RandomCropVideo",
    "RandomResizedCropVideo",
    "CenterCropVideo",
    "ResizeVideo",
    "NormalizeVideo",
    "ToTensorVideo",
    "RandomHorizontalFlipVideo",
    "ColorJitterVideo",
]


class RandomCropVideo(RandomCrop):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, OH, OW)
        """
        i, j, h, w = self.get_params(clip, self.size)
        return F.crop(clip, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomResizedCropVideo(object):
    def __init__(
        self,
        size,
        scale=(0.8, 1.0),
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.scale = scale

    def get_params(self, clip, scale):
        
        H, W = clip.shape[-2:]
        min_dim = min(H, W)

        sampled_size = int(min_dim * random.uniform(scale[0], scale[1]))
        height_offset = random.randint(0, H - sampled_size)
        width_offset = random.randint(0, W - sampled_size)
        return height_offset, width_offset, sampled_size, sampled_size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, H, W)
        """
        i, j, h, w = self.get_params(clip, self.scale)
        return F.resized_crop(clip, i, j, h, w, self.size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(size={0}, interpolation_mode={1}, scale={2})'.format(
                self.size, self.interpolation_mode, self.scale
            )

class CenterResizedCropVideo(RandomResizedCropVideo):
    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        """
        Returns the maximum square, central crop resized to the given image size.
        """
        super(CenterResizedCropVideo, self).__init__(size, scale=1., interpolation_mode=interpolation_mode)

    def get_params(self, clip, scale):

        H, W = clip.shape[-2:]
        min_dim = min(H, W)

        height_offset = int((H - min_dim) // 2)
        width_offset = int((W - min_dim) // 2)

        return height_offset, width_offset, min_dim, min_dim

class CenterCropVideo(object):
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.center_crop(clip, self.crop_size)

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={0})'.format(self.crop_size)

class ResizeVideo(object):
    """Resize the input PIL Images to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.resize_img = Resize(self.size, self.interpolation)

    def __call__(self, imgs):
        """
        Args:
            imgs (List of PIL Image): Images to be scaled.

        Returns:
            List of PIL Images: Rescaled images.
        """
        return [self.resize_img(x) for x in imgs]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)

class NormalizeVideo(object):
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return F.normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(
            self.mean, self.std, self.inplace)


class ToTensorVideo(object):
    """
    Convert a List of PIL Images or numpy ndarrays of each size HxWxC to torch.tensor of size TxCxHxW
    """

    def __init__(self):
        self.totensor = ToTensor()

    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or numpy ndarray): Size is Tx (HxWxC)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        clip = [self.totensor(x) for x in clip]
        return torch.stack(clip)

    def __repr__(self):
        return self.__class__.__name__


class RandomHorizontalFlipVideo(object):
    """
    Flip the video clip along the horizonal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        if random.random() < self.p:
            clip = F.hflip(clip)
        return clip

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)

class ColorJitterVideo(ColorJitter):
    """Randomly change the brightness, contrast and saturation of a list of images.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness, contrast, saturation, hue):

        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def __call__(self, clip):
        """
        Args:
            clip (List): List of 'N' PIL Images.
        Return:
            List of 'N' PIL Images: Color jittered images.
        """

        # transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        # clip = [transform(x) for x in clip]

        clip = [self.forward(x) for x in clip]
        return clip