import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .market1501 import Market1501
from .msmt17 import MSMT17
from .cuhk03 import CUHK03
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist

from PIL import Image
import random
import numpy as np
import albumentations as abm
from collections import deque

from imagecorruptions.corruptions import gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate

import datasets.augmentations.augmentations as augmentations
from .augmentations.augmix import augmix
import math


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


# input PIL, output PIL, put on the top of all transforms
class augmix_transform(object):
    def __init__(self, level=0, width=3, depth=-1, alpha=1.):
        self.level = level
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def __call__(self, img):
        img = augmix(np.asarray(img) / 255)
        img = np.clip(img * 255., 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        return img


def _get_pixels(per_pixel,
                rand_color,
                patch_size,
                dtype=torch.float32,
                device='cuda'):
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


__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'msmt17': MSMT17,
}


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs,
                       dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):

    random_erasing = mixing_erasing(probability=cfg.INPUT.RE_PROB,
                                    mean=cfg.INPUT.PIXEL_MEAN,
                                    type=cfg.INPUT.ERASING_TYPE,
                                    mixing_coeff=cfg.INPUT.MIXING_COEFF)
    re_erasing = mixing_erasing(probability=cfg.INPUT.RE_PROB,
                                mean=cfg.INPUT.PIXEL_MEAN,
                                type='self',
                                mixing_coeff=cfg.INPUT.MIXING_COEFF)

    if cfg.INPUT.AUGMIX:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            augmix_transform(),
            T.ToTensor(), random_erasing, re_erasing
        ])
    else:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(), random_erasing, re_erasing
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
    ])

    val_with_corruption_transforms = T.Compose([
        corruption_transform(0),
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms, cfg=cfg)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(train_set,
                                    batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                    sampler=RandomIdentitySampler(
                                        dataset.train,
                                        cfg.SOLVER.IMS_PER_BATCH,
                                        cfg.DATALOADER.NUM_INSTANCE),
                                    num_workers=num_workers,
                                    collate_fn=train_collate_fn)
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=train_collate_fn)
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.
              format(cfg.SAMPLER))

    train_loader_normal = DataLoader(train_set_normal,
                                     batch_size=cfg.TEST.IMS_PER_BATCH,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     collate_fn=val_collate_fn)



    # val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    query_set = ImageDataset(dataset.query, val_transforms)
    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    corrupted_query_set = ImageDataset(dataset.query, val_with_corruption_transforms)
    corrupted_gallery_set = ImageDataset(dataset.gallery, val_with_corruption_transforms)

    val_set = torch.utils.data.ConcatDataset([query_set, gallery_set])
    corrupted_val_set = torch.utils.data.ConcatDataset([corrupted_query_set, corrupted_gallery_set])
    corrupted_query_set = torch.utils.data.ConcatDataset([corrupted_query_set, gallery_set])
    corrupted_gallery_set = torch.utils.data.ConcatDataset([query_set, corrupted_gallery_set])

    val_loader = DataLoader(val_set,
                            batch_size=cfg.TEST.IMS_PER_BATCH,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=val_collate_fn)
    corrupted_val_loader = DataLoader(corrupted_val_set,
                                      batch_size=cfg.TEST.IMS_PER_BATCH,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      collate_fn=val_collate_fn)
    corrupted_query_loader = DataLoader(corrupted_query_set,
                                        batch_size=cfg.TEST.IMS_PER_BATCH,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        collate_fn=val_collate_fn)
    corrupted_gallery_loader = DataLoader(corrupted_gallery_set,
                                          batch_size=cfg.TEST.IMS_PER_BATCH,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          collate_fn=val_collate_fn)                                   
    return train_loader, train_loader_normal, val_loader, corrupted_val_loader, corrupted_query_loader, corrupted_gallery_loader, len(
        dataset.query), num_classes, cam_num, view_num