# encoding: utf-8

import h5py
import os.path as osp
from scipy.io import loadmat
# imsave are removed since scipy 1.3.0
# from scipy.misc import imsave
from imageio import imsave

from utils.iotools import mkdir_if_missing, write_json, read_json
from .bases import BaseImageDataset
import glob
import re


class CUHK03(BaseImageDataset):
    """
    CUHK03
    Reference:
    Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.
    URL: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!

    Dataset statistics:
    # identities: 1360
    # images: 13164
    # cameras: 6
    # splits: 20 (classic)
    Args:
        split_id (int): split index (default: 0)
        cuhk03_labeled (bool): whether to load labeled images; if false, detected images are loaded (default: False)
    """
    dataset_dir = 'cuhk03'

    def __init__(self,
                 root='./toDataset',
                 cuhk03_labeled=False,
                 verbose=True,
                 **kwargs):
        super(CUHK03, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        print('dataset: ', self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> CUHK03 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(
                self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(
                self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        cam_container = set()
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid, 1))
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        return dataset
