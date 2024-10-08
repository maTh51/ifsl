#!/bin/bash

python main.py --datapath /scratch/dataset/chesapeake/ \
  --benchmark chesapeake \
  --method asnet \
  --logpath asnet \
  --way 5 \
  --shot 1 \
  --bgclass 3 \
  --bsz 1 \
  --fold 0 \
  --backbone resnet50 \
  --eval \
  --bgd \
  # --vis \
  # --rdn_sup \
