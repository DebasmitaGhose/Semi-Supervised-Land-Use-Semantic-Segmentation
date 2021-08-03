import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import pickle
import random
import re
from PIL import ImageOps, ImageFilter


ucm_classes = np.array(('background',  # always index 0
                    'airplane', 'bare_soil', 'buildings', 'cars',
                    'chapparal', 'court', 'dock', 'field', 'grass',
                    'mobile_home', 'pavement', 'sand', 'sea',
                    'ship', 'tanks', 'trees',
                    'water'))
DEEPGLOBE_DATAMAP = '/home/dg777/project/Satellite_Images/DeepGlobeImageSets/class_map.json'

def crete_deepglobe_labels(img_filename, deepglobe_mapping):
    label_ids = list()
    for image in img_filename:
        class_id = deepglobe_mapping[image]
        label_ids.append(class_id)
    return label_ids

def create_ucm_labels(img_filename):
    label_ids = list()
    for image in img_filename:
        print(image)
        arr = re.split('(\d+)', s)
        image_name = arr[0]
        class_id = np.where(ucm_classes==image_name)[0][0]
        label_ids.append(class_id)
    return label_ids
        
class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_folder, img_filename,  dataset, transform=None, train=False):
        self.img_path = os.path.join(data_path, img_folder)
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close(self.img_filename)
        
        if dataset=='UCM':
            label_ids = create_ucm_labels(self.img_filename)
        
        elif dataset=='Deepglobe':
            deepglobe_mapping = json.load(open(DEEPGLOBE_DATAMAP,'r'))
            label_ids = crete_deepglobe_labels(self.img_filename,deepglobe_mapping)
            
        self.create_labels()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        labels = np.loadtxt(label_filepath, dtype=np.int64)
        self.label = label_ids
        self.train = train

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        #if self.transform is not None:
        if self.train:
            img1, img2 = self.transform(img)
        else:
            img = self.transform(img)

        label = torch.from_numpy(self.label[index])
        label = label.type(torch.FloatTensor)
        if self.train:
            return (img1, img2), label
        else:
            return img, label

    def __len__(self):
        return len(self.img_filename)

def split_idxs(pkl_file, percent):

    train_ids = pickle.load(open(pkl_file, 'rb'))
    partial_size = int(percent*len(train_ids))

    labeled_idxs = train_ids[:partial_size]
    unlabeled_idxs = train_ids[partial_size:]

    return labeled_idxs, unlabeled_idxs

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TransformTwice:
    def __init__(self, transform, aug_transform):
        self.transform = transform
        self.aug_transform = aug_transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.aug_transform(inp)
        return out1, out2

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)