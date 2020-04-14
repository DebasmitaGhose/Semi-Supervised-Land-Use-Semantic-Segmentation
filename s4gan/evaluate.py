import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os
import pdb
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from utils.metric import scores
#from model.deeplabv2 import Res_Deeplab
from model import *
#from model.deeplabv3p import Res_Deeplab
from data.voc_dataset import VOCDataSet
from data import get_data_path, get_loader
import torchvision.transforms as transform

from PIL import Image
import scipy.misc


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATASET = 'pascal_voc' # pascal_context

MODEL = 'deeplabv2' # deeeplabv2, deeplabv3p
DATA_DIRECTORY = '../../VOCdevkit/VOC2012/'
DATA_LIST_PATH = '../../VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 21 # 60 for pascal context
RESTORE_FROM = './checkpoints/deeplabv2_resnet101_msc-vocaug-20000.pth'
PRETRAINED_MODEL = None
SAVE_DIRECTORY = 'results'
EXP_ID = "default"
EXP_OUTPUT_DIR = './s4gan_files'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--dataset-split", type=str, default="test",
                        help="train,val,test,subset")
    parser.add_argument("--exp-id", type=str, default=EXP_ID,
                        help= "unique id to identify all files of an experiment")
    parser.add_argument("--check-epoch", type=int, default=10,
                        help="epoch to evaluate the model at")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="dataset name pascal_voc or pascal_context")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory to store results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--with-mlmt", action="store_true",
                        help="combine with Multi-Label Mean Teacher branch")
    parser.add_argument("--save-output-images", action="store_true",
                        help="save output images")
    return parser.parse_args()

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
    vect = hist>0
    vect_out = np.zeros((21,1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0

    return vect_out

def get_iou(args, data_list, class_num, save_path=None):
    from multiprocessing import Pool 
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
 
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    #print(len(m_list[0]))  
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    #print(j_list)

    if args.dataset == 'pascal_voc':
        classes = np.array(('background',  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'))
    elif args.dataset == 'pascal_context':
        classes = np.array(('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bag', 'bed', 'bench', 'book', 'building', 'cabinet' , 'ceiling', 'cloth', 'computer', 'cup',
                'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'keyboard', 'light', 'mountain', 'mouse', 'curtain', 'platform', 'sign', 'plate',
                'road', 'rock', 'shelves', 'sidewalk', 'sky', 'snow', 'bedclothes', 'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood'))
    elif args.dataset == 'cityscapes':
        classes = np.array(("road", "sidewalk",
            "building", "wall", "fence", "pole",
            "traffic_light", "traffic_sign", "vegetation",
            "terrain", "sky", "person", "rider",
            "car", "truck", "bus",
            "train", "motorcycle", "bicycle")) 

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]))
    
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')

def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()
    gpu0 = args.gpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    #pdb.set_trace()
    #model = Res_Deeplab(num_classes=args.num_classes)
    model = DeepLabV2_ResNet101_MSC(n_classes=args.num_classes)
    model.cuda()

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    elif EXP_ID  == "default" :
        saved_state_dict = torch.load(os.path.join(RESTORE_FROM))
    else:
        #saved_state_dict = torch.load(args.restore_from)
        print(os.path.join(EXP_OUTPUT_DIR, "models", args.exp_id, "train", 'checkpoint'+str(args.check_epoch)+'.pth'), 'saved weights')
        saved_state_dict = torch.load(os.path.join(EXP_OUTPUT_DIR, "models", args.exp_id, "train", 'checkpoint'+str(args.check_epoch)+'.pth'))
    #print("Restoring from: ", saved_state_dict)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    if args.dataset == 'pascal_voc':
        testloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, crop_size=(320, 240), mean=IMG_MEAN, scale=False, mirror=False), 
                                    batch_size=1, shuffle=False, pin_memory=True)
        interp = nn.Upsample(size=(320, 240), mode='bilinear', align_corners=False) # align corners = True

    elif args.dataset == 'pascal_context':
        input_transform = transform.Compose([transform.ToTensor(),
                transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        data_kwargs = {'transform': input_transform, 'base_size': 512, 'crop_size': 512}
        data_loader = get_loader('pascal_context')
        data_path = get_data_path('pascal_context')
        test_dataset = data_loader(data_path, split='val', mode='val', **data_kwargs)
        testloader = data.DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=1, pin_memory=True)
        interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    elif args.dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        test_dataset = data_loader( data_path, img_size=(512, 1024), is_transform=True, split='val')
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)
    
    data_list = []
    gt_list = []
    output_list = []
    colorize = VOCColorize()
   
    if args.with_mlmt:
        mlmt_preds = np.loadtxt('mlmt_output/output_ema_p_1_0_voc_5.txt', dtype = float) # best mt 0.05

        mlmt_preds[mlmt_preds>=0.2] = 1
        mlmt_preds[mlmt_preds<0.2] = 0 
 
    for index, batch in enumerate(testloader):
        if index % 1 == 0:
            print('%d processd'%(index))
        image, label, size, name, _ = batch
        size = size[0]
        output  = model(Variable(image, volatile=True).cuda(gpu0))
        output = interp(output).cpu().data[0].numpy()

        if args.dataset == 'pascal_voc':
            output = output[:,:size[0],:size[1]]
            gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
        elif args.dataset == 'pascal_context':
            gt = np.asarray(label[0].numpy(), dtype=np.int)
        elif args.dataset == 'cityscapes':
            gt = np.asarray(label[0].numpy(), dtype=np.int)

        if args.with_mlmt:
            for i in range(args.num_classes):
                output[i]= output[i]*mlmt_preds[index][i]
        
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

        viz_dir = os.path.join(EXP_OUTPUT_DIR, "output_viz", args.exp_id, args.dataset_split)
        
        if not os.path.exists(viz_dir):
            makedirs(viz_dir)
        print("Visualization dst:", viz_dir)
        
        if args.save_output_images:
            if args.dataset == 'pascal_voc':
                filename = '{}.png'.format(name[0])
                color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
                color_file.save(os.path.join(viz_dir, filename))
            #elif args.dataset == 'pascal_context':
            #    filename = os.path.join(args.save_dir, filename[0])
            #    scipy.misc.imsave(filename, gt)
        
        data_list.append([gt.flatten(), output.flatten()])
        gt_list.append(gt)
        output_list.append(output)
        #score = scores(data_list[0], output.flatten(), args.num_classes)
        #print(score)
    #print(np.shape(data_list[0][:]),'data list')
    scores_dir = os.path.join(EXP_OUTPUT_DIR, "scores", args.exp_id, args.dataset_split)
    if not os.path.exists(scores_dir):
        makedirs(scores_dir)
    scores_filename = os.path.join(scores_dir, "scores.json")
    print("Scores saved at: ",scores_filename)

    

        #get_iou(args, data_list, args.num_classes, filename)
    #print(np.shape(gt), 'gt')
    #print(gt==output, 'gt')
    #print(np.shape(output), 'output')
    #print(output, 'output')
    #print(np.shape(gt_list[0]), 'gt list')
    #print(np.shape(output_list[0]), 'output list')
    score = scores(gt_list, output_list, args.num_classes)
    #print(score)
    with open(scores_filename, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
