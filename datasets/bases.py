from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
from .augmentations.augmix import augmix
import torchvision.transforms as T
import numpy as np
from timm.data.random_erasing import RandomErasing
import math

from PIL import Image
import random
import numpy as np
import albumentations as abm
from collections import deque

from imagecorruptions.corruptions import gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate

import hashlib


def rain(image, severity=1):
    if severity == 1:
        type = 'drizzle'
    elif severity == 2 or severity == 3:
        type = 'heavy'
    elif severity == 4 or severity == 5:
        type = 'torrential'
    blur_value = 2 + severity
    bright_value = -(0.05 + 0.05 * severity)
    rain = abm.Compose([
        abm.augmentations.transforms.RandomRain(rain_type=type,
                                                blur_value=blur_value,
                                                brightness_coefficient=1,
                                                always_apply=True),
        abm.augmentations.transforms.RandomBrightness(
            limit=[bright_value, bright_value], always_apply=True)
    ])
    width, height = image.size
    if height <= 60:
        scale_factor = 65.0 / height
        new_size = (int(width * scale_factor), 65)
        image = image.resize(new_size)
    return rain(image=np.array(image))['image']


corruption_function = [
    gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur,
    motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
    elastic_transform, pixelate, jpeg_compression, speckle_noise,
    gaussian_blur, spatter, saturate, rain
]


class corruption_transform(object):
    def __init__(self, level=0, type='all'):
        self.level = level
        self.type = type

    def __call__(self, img):
        if self.level > 0 and self.level < 6:
            level_idx = self.level
        else:
            level_idx = random.choice(range(1, 6))
        if self.type == 'all':
            corrupt_func = random.choice(corruption_function)
        else:
            func_name_list = [f.__name__ for f in corruption_function]
            corrupt_idx = func_name_list.index(self.type)
            corrupt_func = corruption_function[corrupt_idx]
        c_img = corrupt_func(img.copy(), severity=level_idx)
        img = Image.fromarray(np.uint8(c_img))
        return img


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print(
                "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill."
                .format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """
    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """
    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(
            train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(
            query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(
            gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(
            num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(
            num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(
            num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


def _get_pixels(per_pixel,
                rand_color,
                patch_size,
                dtype=torch.float32,
                device='cuda',
                mean=(0.5, 0.5, 0.5)):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype,
                           device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class mixing_erasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels with different mixing operation.
    normal: original random erasing;
    soft: mixing ori with random pixel;
    self: mixing ori with other_ori_patch;
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self,
                 probability=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=(0.4914, 0.4822, 0.4465),
                 mode='pixel',
                 device='cpu',
                 type='normal',
                 mixing_coeff=[1.0, 1.0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.rand_color = False
        self.per_pixel = False
        self.mode = mode
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device
        self.type = type
        self.mixing_coeff = mixing_coeff

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if self.type == 'normal':
                    m = 1.0
                else:
                    m = np.float32(
                        np.random.beta(self.mixing_coeff[0],
                                       self.mixing_coeff[1]))
                if self.type == 'self':
                    x2 = random.randint(0, img.size()[1] - h)
                    y2 = random.randint(0, img.size()[2] - w)
                    img[:, x1:x1 + h,
                        y1:y1 + w] = (1 - m) * img[:, x1:x1 + h, y1:y1 +
                                                   w] + m * img[:, x2:x2 + h,
                                                                y2:y2 + w]
                else:
                    if self.mode == 'const':
                        img[0, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[0, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[0]
                        img[1, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[1, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[1]
                        img[2, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[2, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[2]
                    else:
                        img[:, x1:x1 + h, y1:y1 +
                            w] = (1 - m) * img[:, x1:x1 + h,
                                               y1:y1 + w] + m * _get_pixels(
                                                   self.per_pixel,
                                                   self.rand_color,
                                                   (img.size()[0], h, w),
                                                   dtype=img.dtype,
                                                   device=self.device)
                return img
        return img


class RandomPatch(object):
    """Random patch data augmentation.
    There is a patch pool that stores randomly extracted pathces from person images.
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """
    def __init__(
        self,
        prob_happen=0.5,
        pool_capacity=50000,
        min_sample_size=100,
        patch_min_area=0.01,
        patch_max_area=0.5,
        patch_min_ratio=0.1,
        prob_flip_leftright=0.5,
    ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area,
                                         self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio,
                                          1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = torch.flip(patch, dims=[2])
        return patch

    def __call__(self, img):
        _, H, W = img.size()  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img[..., y1:y1 + h, x1:x1 + w]
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        _, patchH, patchW = patch.size()
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img[..., y1:y1 + patchH, x1:x1 + patchW] = patch

        return img



class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, cfg=None):
        self.dataset = dataset
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.cfg is not None and self.cfg.INPUT.SELF_ID:
            random_erasing = mixing_erasing(
                probability=self.cfg.INPUT.RE_PROB,
                mean=self.cfg.INPUT.PIXEL_MEAN,
                type=self.cfg.INPUT.ERASING_TYPE,
                mixing_coeff=self.cfg.INPUT.MIXING_COEFF)
            re_erasing = mixing_erasing(
                probability=self.cfg.INPUT.RE_PROB,
                mean=self.cfg.INPUT.PIXEL_MEAN,
                type='self',
                mixing_coeff=self.cfg.INPUT.MIXING_COEFF)
            pre_transform = T.Compose([
                T.Resize(self.cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=self.cfg.INPUT.PROB),
                T.Pad(self.cfg.INPUT.PADDING),
                T.RandomCrop(self.cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(), random_erasing, re_erasing
            ])
            
            post_transform = T.Compose([
                T.ToTensor(),
            ])

            img = pre_transform(img)

            img = T.ToPILImage()(img).convert('RGB')
            if self.cfg.INPUT.AUGMIX:
                img = np.asarray(img) / 255.
                img1 = augmix(img)
                img2 = augmix(img)
                img1 = np.clip(img1 * 255., 0, 255).astype(np.uint8)
                img2 = np.clip(img2 * 255., 0, 255).astype(np.uint8)

            img = post_transform(img)
            img1 = post_transform(img1)
            img2 = post_transform(img2)

            img_list = [img, img1, img2]
            img_tuple = torch.cat(img_list, 0).half()
            return img_tuple, pid, camid, trackid, img_path.split('/')[-1]
        else:

            if self.transform is not None:
                img = self.transform(img)

            return img, pid, camid, trackid, img_path.split('/')[-1]