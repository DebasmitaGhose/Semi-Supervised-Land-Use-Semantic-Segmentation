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
                    'agricultural', 'airplane', 'baseballdiamond', 'beach','buildings',
                    'chaparral',  'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 
                    'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 
                    'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt'))
DEEPGLOBE_DATAMAP = '/home/dg777/project/Satellite_Images/DeepGlobeImageSets/class_map.json'

def create_one_hot_encoding(label_id):
    labels_arr = np.zeros(21)
    labels_arr[label_id-1] = 1
    return labels_arr

def create_deepglobe_labels(img_filename, deepglobe_mapping):
    label_ids = list()
    for image in img_filename:
        class_id = deepglobe_mapping[image]
        label_ids.append(class_id)
    return label_ids

def create_ucm_labels(img_filename):
    label_ids = list()
    for image in img_filename:
        print(image)
        arr = re.split('(\d+)', image)
        print(arr, np.where(ucm_classes==arr[0]))
        image_name = arr[0]
        class_id = np.where(ucm_classes==image_name)[0][0]
        label_ids.append(class_id)
    return label_ids
        
class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_folder, img_filename,  dataset_name, transform=None, train=False):
        self.img_path = os.path.join(data_path, img_folder)
        self.transform = transform
        label_ids = list()
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        self.img_filename = [image_name + '.tif' for image_name in self.img_filename]
        fp.close()
        #print('dataset_name', dataset_name, self.img_path)
        if dataset_name =='UCM':
            label_ids = create_ucm_labels(self.img_filename)
        
        elif dataset_name =='Deepglobe':
            deepglobe_mapping = json.load(open(DEEPGLOBE_DATAMAP,'r'))
            label_ids = create_deepglobe_labels(self.img_filename,deepglobe_mapping)
        #print(label_ids)    
        self.label = label_ids
        self.train = train

    def __getitem__(self, index):
        #print(os.path.join(self.img_path, self.img_filename[index]))
        
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        self.transform(img)
        #import pdb;pdb.set_trace()
        #print('getting item')
        #print(index, self.img_filename[index] ,self.label[index],  img)
        #if self.transform is not None:
        if self.train:
            #if img is not None:
            #    print(index)
            #try:  
            img1, img2 = self.transform(img)
            #except Exception as e:
            #    print(index, e) 
        else:
            img = self.transform(img)
        one_hot_label = create_one_hot_encoding(self.label[index])
        label = torch.from_numpy(np.array(one_hot_label))
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
