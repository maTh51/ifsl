#!/bin/bash

python main.py --datapath /scratch/matheuspimenta/ \
  --benchmark vaihingen \
  --method panet \
  --logpath panet \
  --way 4 \
  --shot 5 \
  --bsz 1 \
  --fold 0 \
  --backbone resnet50 \
  --eval \
  --bgd \
  --rdn_sup \
  # --vis \
