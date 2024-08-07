#!/bin/bash

python main.py --datapath /scratch/gabrielamarante/dataset/ \
  --benchmark chesapeake \
  --method panet \
  --logpath panet \
  --way 6 \
  --shot 1 \
  --bsz 1 \
  --fold 0 \
  --backbone resnet50 \
  --eval \
  --bgd \
  # --vis \
  # --rdn_sup \
