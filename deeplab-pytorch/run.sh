#!/bin/bash
#SBATCH --job-name=DeepLab_VOC
#SBATCH --ntasks=2 --nodes=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dghose@cs.umass.edu

#python main.py test --config-path configs/voc12.yaml --model-path deeplabv2_resnet101_msc-vocaug-20000.pth --cuda

python main.py train --config-path configs/voc12.yaml --cuda

