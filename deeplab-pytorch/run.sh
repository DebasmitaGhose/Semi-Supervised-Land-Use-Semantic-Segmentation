#!/bin/bash
#SBATCH --job-name=DeepLab_VOC
#SBATCH --ntasks=6 --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dghose@cs.umass.edu

python main.py test --config-path configs/voc12.yaml --model-path deeplabv2_resnet101_msc-vocaug-20000.pth --cuda
