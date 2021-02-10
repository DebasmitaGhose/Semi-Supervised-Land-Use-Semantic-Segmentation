#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   08 February 2019

from __future__ import absolute_import, print_function

import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .base import _BaseDataset


class DeepGlobe(_BaseDataset):
    """
    UC Merced Segmentation dataset
    """

    def __init__(self, **kwargs):
        super(DeepGlobe, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = osp.join(self.root, "Satellite_Images/DeepGlobe_Images/") # changed dataset here
        self.label_dir = osp.join(self.root, "Satellite_Images/DeepGlobe_Labels/") # changed label here

        if self.split in ["all_images", "train_images", "test_images"]: # added subset here
            file_list = osp.join(self.root,'Satellite_Images/DeepGlobeImageSets' , self.split + ".txt")
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.root, self.image_dir, image_id + "_sat.jpg")
        label_path = osp.join(self.root, self.label_dir, image_id + "_mask.png")
        # Load an image
        #print(label_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.resize(image, (320,320), interpolation=cv2.INTER_CUBIC)
        #image = cv2.imread(image_path, -1).astype(np.float32) # https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images
        #label = np.asarray(Image.open(label_path), dtype=np.int32)
        label = cv2.imread(label_path)
        #print(label.shape)
        label = cv2.resize(label, (320,320), interpolation=cv2.INTER_CUBIC)
        label = self.encode_segmap(label)
        #print(label.shape)
        return image_id, image, label
    
    def get_deepglobe_labels(self):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (7, 3)
        """
        return np.asarray(
            [
                [0, 255, 255],#urban_land
                [255, 255, 0], #agriculture_land	
                [255, 0, 255], #rangeland
                [0, 255, 0], #forest 
                [0, 0, 255], #water 
                [255, 255, 255],#barren  
                [0, 0, 0] #unknown
            ]
        ) 

    #https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py#L140
    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_deepglobe_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        #label_mask = label_mask.astype(int)
        return label_mask         



if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import torchvision
    import yaml
    from torchvision.utils import make_grid
    from tqdm import tqdm

    kwargs = {"nrow": 10, "padding": 50}
    batch_size = 100

    dataset = VOCAug(
        root="/media/kazuto1011/Extra/VOCdevkit",
        split="train", #train_aug
        ignore_label=255,
        mean_bgr=(104.008, 116.669, 122.675),
        year=2012,
        augment=True,
        base_size=None,
        crop_size=513,
        scales=(0.5, 0.75, 1.0, 1.25, 1.5),
        flip=True,
    )
    print(dataset)

    loader = data.DataLoader(dataset, batch_size=batch_size)

    for i, (image_ids, images, labels) in tqdm(
        enumerate(loader), total=np.ceil(len(dataset) / batch_size), leave=False
    ):
        if i == 0:
            mean = torch.tensor((104.008, 116.669, 122.675))[None, :, None, None]
            images += mean.expand_as(images)
            image = make_grid(images, pad_value=-1, **kwargs).numpy()
            image = np.transpose(image, (1, 2, 0))
            mask = np.zeros(image.shape[:2])
            mask[(image != -1)[..., 0]] = 255
            image = np.dstack((image, mask)).astype(np.uint8)

            labels = labels[:, np.newaxis, ...]
            label = make_grid(labels, pad_value=255, **kwargs).numpy()
            label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
            label = cm.jet_r(label_ / 21.0) * 255
            mask = np.zeros(label.shape[:2])
            label[..., 3][(label_ == 255)] = 0
            label = label.astype(np.uint8)

            tiled_images = np.hstack((image, label))
            # cv2.imwrite("./docs/datasets/voc12.png", tiled_images)
            plt.imshow(np.dstack((tiled_images[..., 2::-1], tiled_images[..., 3])))
            plt.show()
            break
