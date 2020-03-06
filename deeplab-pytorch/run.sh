#!/bin/bash
#SBATCH --job-name=evalDeepLab_VOC
#SBATCH --ntasks=2 --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH -o run_logs/evalDeepLab_VOC_%j.txt
#SBATCH -e run_logs/evalDeepLab_VOC_%j.err
#SBATCH --mail-user=dghose@cs.umass.edu

#python main.py test --config-path configs/voc12.yaml --model-path deeplabv2_resnet101_msc-vocaug-20000.pth --cuda

#python main.py train --config-path configs/voc12.yaml --cuda

#python main.py train --config-path configs/ucm.yaml --cuda

python main.py test --config-path configs/ucm.yaml --model-path data/models/ucm1/deeplabv2_resnet101_msc/train/checkpoint_final.pth --cuda

