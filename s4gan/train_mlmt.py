import argparse
import os
import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import data.dataset_processing as data
from data.dataset_processing import TransformTwice, GaussianBlur, update_ema_variables

global_step = 0

IMAGE_FOLDER = 'UCMerced_Images/'
DATA_LIST = '/home/dg777/project/Satellite_Images/UCMImageSets/train.txt'
RANDOM_SEED = 1234
SAMPLING_TYPE = "uncertainty"
EXP_OUTPUT_DIR = './mlmt_files'
SAVE_PRED_EVERY=  1
#DATA_LIST = '/home/dg777/project/Satellite_Images/DeepGlobeImageSets/train_images.txt'




#TRAIN_DATA = 'train'
#TEST_DATA = 'val'
#TRAIN_IMG_FILE = 'train_img.txt'
#TEST_IMG_FILE = 'val_img.txt'
#TRAIN_LABEL_FILE = 'train_label.txt'
#TEST_LABEL_FILE = 'val_label.txt'

m = nn.Sigmoid()

def get_arguments():
    parser = argparse.ArgumentParser(description="MLMT Network Branch")
    parser.add_argument("--lr", type=float, default=3e-2, help="learning rate")
    parser.add_argument("--eta-min", type=float, default=1e-4, help="minimum learning rate for the scheduler")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="optimizer: weight decay")
    parser.add_argument("--workers", type=int, default=4, help="number of workers")
    parser.add_argument("--num-classes", type=int, default=21, help="number of classes, For eg 21 in VOC")

    parser.add_argument("--batch-size-lab", type=int, default=1, help="minibatch size of labeled training set")#16
    parser.add_argument("--batch-size-unlab", type=int, default=1, help="minibatch size of unlabeled training set")#80
    parser.add_argument("--batch-size-val", type=int, default=1, help="minibatch size of validation set")#32

    parser.add_argument("--num-epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--burn-in-epochs", type=int, default=10, help="number of burn-in epochs")
    parser.add_argument("--evaluation-epochs", type=int, default=0, help="evaluation epochs")

    parser.add_argument('--exp-name', type=str, default='default', help="experiment name")
    parser.add_argument('--cons-loss', type=str, default='cosine', help="consistency loss type: cosine")
    parser.add_argument('--data-dir', type=str, default='./data/voc_dataset/', help="dataset directory path")
    parser.add_argument('--pkl-file', type=str, default='./checkpoints/voc_semi_0_125/train_voc_split.pkl', help="indexes of files")

    parser.add_argument("--w-cons", type=float, default=1.0, help="weightage consistency loss term")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="decay rate of exponential moving average")
    parser.add_argument("--labeled-ratio", type=float, default=0.125, help="percent of labeled samples")
    parser.add_argument('--verbose', action='store_true', help='verbose')

    # new  arguments
    parser.add_argument('--exp-id', type=str, default='default', help="experiment id")
    parser.add_argument("--evaluate", action='store_true', default=False, help="whether to evaluate or not")
    parser.add_argument("--image-folder", type = str, default = IMAGE_FOLDER, help = "Path to the image  folder.")
    parser.add_argument("--image-list", type = str, default = DATA_LIST,
                       help = "Path to the file listing the images in the dataset.")
    parser.add_argument("--dataset-name", type = str, help = "UCM or Deepglobe", default='Deepglobe')
    parser.add_argument("--active-learning", action='store_true', default=False)
    parser.add_argument("--sampling-type", type=str, default=SAMPLING_TYPE,
                        help="sampling technique to use")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,help="random seed")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--dataset-split", type=str, default='train',
                        help="train or test")
    return parser.parse_args()

args = get_arguments()

np.random.seed(args.random_seed)

if args.verbose:
    from utils.visualize import progress_bar

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def main():
    global global_step

   
    checkpoint_dir = os.path.join(
        EXP_OUTPUT_DIR,
        "models",
        args.exp_id,
        args.dataset_split,
        str(args.labeled_ratio))
    makedirs(checkpoint_dir)
    #train_loader_lab, train_loader_unlab, valloader = create_data_loaders()
    train_loader_lab, train_loader_unlab = create_data_loaders()

    print ('data loaders ready !!')

    def create_model(ema=False):
        model = models.resnet101(pretrained=True)
        model.fc = nn.Linear(2048, args.num_classes)
        model = torch.nn.DataParallel(model)
        model.cuda()
        cudnn.benchmark = True

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    model_mt = create_model(ema=True)

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=args.num_epochs,
                                                        eta_min=args.eta_min)

    for epoch in range(args.num_epochs):
        print ('Epoch#: ', epoch)

        train(train_loader_lab, train_loader_unlab, model, model_mt, optimizer, epoch, checkpoint_dir)
        scheduler.step()

        if args.evaluate and args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            print ("Evaluating the primary model:")
            validate(valloader, 'val', model, epoch + 1)
            print ("Evaluating the MT model:")
            validate(valloader, 'ema', model_mt, epoch + 1)

def create_data_loaders():
    channel_stats = dict(mean=[.485, .456, .406],
                         std=[.229, .224, .225])

    #channel_stats = dict

    transform_train = transforms.Compose([
        transforms.Resize(size=(320, 320), interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    transform_aug = transforms.Compose([
        transforms.Resize(size=(320, 320), interpolation=2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomHorizontalFlip(),
        GaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    '''
    transform_test = transforms.Compose([
        transforms.Resize(size=(320, 320), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    '''
    
    transform_lab = TransformTwice(transform_train, transform_train)
    transform_unlab = TransformTwice(transform_train, transform_aug)


 
    print ('loading data ...')
    dataset = data.DatasetProcessing(
        args.data_dir, args.image_folder, args.image_list, args.dataset_name, transform_lab, train=True)

    dataset_aug = data.DatasetProcessing(
        args.data_dir, args.image_folder, args.image_list, args.dataset_name, transform_unlab,  train=True)

    train_dataset_size = len(dataset)    

    if args.active_learning:
        active_list_path =  args.sampling_type + '/' + args.dataset_name + '_' + args.sampling_type + '_' + str(args.labeled_ratio) + '.txt'
        #print(active_list_path, 'active list path')
        active_img_names = [i_id.strip() for i_id in open(active_list_path)]
        #print(np.shape(active_img_names), 'active image names')
        all_img_names = [i_id.strip() for i_id in open(args.image_list)]
        #print(all_img_names, 'all image names')
    
        active_img_names = np.array(active_img_names)
        all_img_names = np.array(all_img_names)

        '''
        numpy.isin(element, test_elements, assume_unique=False, invert=False)
        Calculates element in test_elements, broadcasting over element only.
        Returns a boolean array of the same shape as element that is True
        where an element of element is in test_elements and False otherwise.
        '''
        active_ids = np.where(np.isin(all_img_names, active_img_names))  # np.isin will return a boolean array of size all_image_names.
        #print(active_ids, 'active ids')
        active_ids = active_ids[0]
        #print(np.shape(active_ids), 'active ids')

        train_ids = np.arange(train_dataset_size)
        #print(train_ids, 'train_ids')
        remaining_ids = np.delete(train_ids, active_ids)
        sampler_lab = SubsetRandomSampler(active_ids)
        sampler_unlab = SubsetRandomSampler(remaining_ids)
        
    
        print ('number of labeled samples: ', len(active_ids))
        print ('number of unlabeled samples: ', len(remaining_ids))

    else:
        partial_size = int(args.labeled_ratio * train_dataset_size)

        print(partial_size, "partial size")
        train_ids = np.arange(train_dataset_size)
        # print(train_ids, "train ids")
        np.random.shuffle(train_ids)
        sampler_lab = SubsetRandomSampler(train_ids[:partial_size])
        sampler_unlab = SubsetRandomSampler(train_ids[partial_size:])

    trainloader_lab = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size_lab,
                                               sampler=sampler_lab,
                                               num_workers=args.workers,
                                               pin_memory=True)

    trainloader_unlab = torch.utils.data.DataLoader(dataset_aug,
                                               batch_size=args.batch_size_unlab,
                                               sampler=sampler_unlab,
                                               num_workers=args.workers,
                                               pin_memory=True)

    #dataset_test = data.DatasetProcessing(
    #    args.data_dir, TEST_DATA, TEST_IMG_FILE, TEST_LABEL_FILE, transform_test, train=False)

    '''
    valloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=2 * args.workers,
        pin_memory=True,
        drop_last=False)
    '''
    #return trainloader_lab, trainloader_unlab, valloader
    return trainloader_lab, trainloader_unlab


def cosine_loss(p_logits, q_logits):
    return torch.nn.CosineEmbeddingLoss()(q_logits, p_logits.detach(), torch.ones(p_logits.shape[0]).cuda())

def train(trainloader_lab, trainloader_unlab, model, model_mt, optimizer, epoch, checkpoint_dir):
    global global_step

    loss_sum = 0.0
    class_loss_sum = 0.0
    cons_loss_sum = 0.0
    avg_acc_sum = 0.0
    avg_acc_sum_mt = 0.0

    class_criterion = nn.BCELoss().cuda()

    # switch to train mode
    model.train()
    model_mt.train()

    trainloader_unlab_iter = iter(trainloader_unlab)

    for batch_idx, ((inputs, _), target) in enumerate(trainloader_lab):

        #target = target.squeeze(2).float()
        inputs, target = inputs.cuda(), target.cuda()

        model_out = m(model(inputs))
        model_mt_out = m(model_mt(inputs))
        class_loss = class_criterion(model_out, target)

        try:
            batch_unlab = next(trainloader_unlab_iter)
        except:
            trainloader_unlab_iter = iter(trainloader_unlab)
            batch_unlab = next(trainloader_unlab_iter)

        (inputs_unlab, inputs_unlab_aug), _ = batch_unlab
        inputs_unlab, inputs_unlab_aug = inputs_unlab.cuda(), inputs_unlab_aug.cuda()

        model_unlab_out_aug = model(inputs_unlab_aug)
        with torch.no_grad():
            model_mt_unlab_out = model_mt(inputs_unlab)

        cons_loss = cosine_loss(model_mt_unlab_out, model_unlab_out_aug)

        if epoch>args.burn_in_epochs:
            w_cons = min(args.w_cons, (epoch-args.burn_in_epochs)*2/args.num_epochs)
        else:
            w_cons = 0.0

        loss = class_loss + w_cons*cons_loss
 
        class_loss_sum += class_loss.item()
        cons_loss_sum += cons_loss.item()
        loss_sum += loss.item()

        avg_acc, acc_zeros, acc_ones, acc = accuracy(model_out, target)
        avg_acc_mt, acc_zeros_mt, acc_ones_mt, acc_mt = accuracy(model_mt_out, target)

        avg_acc_sum += avg_acc
        avg_acc_sum_mt += avg_acc_mt

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, model_mt, args.ema_decay, global_step)

        if epoch % args.save_pred_every == 0 and epoch!=0:       
            torch.save(model.module.state_dict(),os.path.join(checkpoint_dir, 'checkpoint'+str(epoch)+'.pth'))
            torch.save(model_mt.module.state_dict(),os.path.join(checkpoint_dir, 'checkpoint'+str(epoch)+'_mt.pth'))
   
        if args.verbose:
            progress_bar(batch_idx, len(trainloader_lab), 'Loss: %.3f |  Class Loss: %.3f |  Cons Loss: %.3f | Avg Acc: %.3f | Avg Acc MT: %.3f '
                % (loss_sum/(batch_idx+1), class_loss_sum/(batch_idx+1), cons_loss_sum/(batch_idx+1), avg_acc_sum/(batch_idx+1), avg_acc_sum_mt/(batch_idx+1)))
    if not args.verbose:
        print('Loss: ', loss_sum/(batch_idx+1), ' Class Loss: ', class_loss_sum/(batch_idx+1),  ' Cons Loss: ', cons_loss_sum/(batch_idx+1), ' Avg Acc: ', avg_acc_sum/(batch_idx+1), ' Avg Acc MT: ', avg_acc_sum_mt/(batch_idx+1))


def validate(eval_loader, mode, model, epoch):

    avg_acc_sum = 0.0
    ones_acc_sum = 0.0
    zeros_acc_sum = 0.0

    if mode=='val':
        filename_raw = 'output_val_raw_' + str(epoch) + '.txt'
        filename_bin = 'output_val_bin_' + str(epoch) + '.txt'
    if mode == 'ema':
        filename_raw = 'output_ema_raw_' + str(epoch) + '.txt'
        filename_bin = 'output_ema_bin_' + str(epoch) + '.txt'

    mlmt_output_path = os.path.join('./mlmt_output', args.exp_name)

    if not os.path.exists(mlmt_output_path):
        os.makedirs(mlmt_output_path)

    f_raw = open(os.path.join(mlmt_output_path, filename_raw), 'a')
    f_bin = open(os.path.join(mlmt_output_path, filename_bin), 'a')

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(eval_loader):

            inputs, target = inputs.cuda(), target.cuda()

            # compute output
            output = m(model(inputs))

            if epoch%1 == 0:
                output_raw = output.cpu().numpy()
                output_raw = np.roll(output_raw, 1)
                output_bin = (output_raw>0.5)*1
                np.savetxt(f_raw, output_raw, fmt='%f')
                np.savetxt(f_bin, output_bin, fmt='%d')

            # measure accuracy and record loss
            avg_acc, acc_zeros, acc_ones, acc = accuracy(output, target)

            ones_acc_sum += acc_ones
            zeros_acc_sum += acc_zeros
            avg_acc_sum += avg_acc
            if args.verbose:
                progress_bar(batch_idx, len(eval_loader), '| Avg Acc: %.3f | Ones Acc: %.3f | Zeros Acc: %.3f |'
                    % (avg_acc_sum/(batch_idx+1), ones_acc_sum/(batch_idx+1), zeros_acc_sum/(batch_idx+1)))
        if not args.verbose:
            print(batch_idx, len(eval_loader), ' Avg Acc: ', avg_acc_sum/(batch_idx+1))

    f_raw.close()
    f_bin.close()

def accuracy(outputs, targets):
    thres = torch.ones(targets.size(0), args.num_classes)*0.5
    thres = thres.cuda()

    cond = torch.ge(outputs, thres)

    count_label_ones = 0
    count_label_zeros = 0
    correct_ones = 0
    correct_zeros = 0
    correct = 0
    total = 0

    for i in range(targets.size(0)):
        for  j in range(args.num_classes):
            if targets[i][j]==0:
                count_label_zeros +=1
            if targets[i][j]==1:
                count_label_ones +=1

    targets = targets.type(torch.ByteTensor).cuda()

    for i in range(targets.size(0)):
        for  j in range(args.num_classes):
            if targets[i][j]==cond[i][j]:
                correct +=1
                if targets[i][j] == 0:
                    correct_zeros +=1
                elif targets[i][j] ==1:
                    correct_ones +=1
    total += targets.size(0)*args.num_classes

    total_acc = (correct_zeros + correct_ones)*100.0/total
    avg_acc = (correct_ones/count_label_ones + correct_zeros/count_label_zeros)*100.0/2.0
    acc_zeros = (100.*correct_zeros/count_label_zeros)
    acc_ones =  (100.*correct_ones/count_label_ones)

    return avg_acc, acc_zeros, acc_ones, total_acc

if __name__ == '__main__':
    main()
