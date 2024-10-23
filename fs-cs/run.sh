#!/bin/bash

# Criando pasta (se não existe)
mkdir -p experiments

# Parâmetros fixos
BASE_DATAPATH="/scratch/dataset"
BSZ=1
FOLD=0
BACKBONE="resnet50"
EVAL="--eval"
# WAY = 5

# Parâmetros variáveis
METHODS=("panet" "pfenet" "hsnet" "asnet")
SHOTS=(2 5)                   
BGCLASSES=(3)                        
# BENCHMARKS=("chesapeake" "vaihingen")
BENCHMARKS=("chesapeake")  

for BENCHMARK in "${BENCHMARKS[@]}"; do
  DATAPATH="${BASE_DATAPATH}/${BENCHMARK}"
  if [ "$BENCHMARK" = "chesapeake" ]; then
    WAY=5
  elif [ "$BENCHMARK" = "vaihingen" ]; then
    WAY=4
  fi

  for METHOD in "${METHODS[@]}"; do
    for SHOT in "${SHOTS[@]}"; do
      for BGCLASS in "${BGCLASSES[@]}"; do
        LOGFILE="experiments/chesapeake-testset/output_${BENCHMARK}_${METHOD}_shot${SHOT}_bgclass${BGCLASS}_$(date +%Y%m%d_%H%M%S).log"

        echo "Rodando com benchmark=$BENCHMARK, method=$METHOD, shot=$SHOT, bgclass=$BGCLASS"

        python main.py --datapath $DATAPATH \
          --benchmark $BENCHMARK \
          --method $METHOD \
          --logpath $METHOD \
          --way $WAY \
          --shot $SHOT \
          --bgclass $BGCLASS \
          --bsz $BSZ \
          --fold $FOLD \
          --backbone $BACKBONE \
          --bgd \
          $EVAL > $LOGFILE 2>&1
      done
    done
  done
done