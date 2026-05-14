#!/usr/bin/env bash
set -euo pipefail

mkdir -p experiments

# ---------------------------------
# General configuration
# ---------------------------------
MODE="eval"                        # train | eval
BASE_DATAPATH="/scratch/dataset"
OEM_DATAPATH="/scratch/matheuspimenta/oem"

METHOD="pfenet"                    # panet | pfenet | hsnet | asnet | asnethm
BACKBONE="resnet101"
FOLD=0
GPU=1

USE_BGD=true
USE_NOWANDB=true

# ---------------------------------
# Train configuration
# ---------------------------------
TRAIN_BENCHMARK="oem"              # chesapeake | vaihingen | oem | pascal | coco
TRAIN_SHOT=1
TRAIN_BGCLASS=0
TRAIN_BSZ=64
TRAIN_NITER=200

# ---------------------------------
# Eval configuration (cross-dataset)
# ---------------------------------
EVAL_METHODS=("panet" "hsnet" "asnet" "pfenet")
# Optional override for checkpoint location:
# - Empty: auto-resolve as logs/pascal/fold{FOLD}/{BACKBONE}/{method}
# - Fixed path: use same checkpoint for all methods
# - Template path: can use {method} placeholder
#   Example: logs/pascal/fold0/resnet101/{method}
CKPTPATH=""
EVAL_BENCHMARKS=("oem")
EVAL_WAYS=(1 2)
EVAL_SHOTS=(1 5)
EVAL_BGCLASSES=(0)
EVAL_BSZ=16
EVAL_SUPPORT_STRATEGIES=("random" "similarity" "max_area")
EVAL_SUPPORT_AREA_CACHE="auto"             # auto | relative path in pools | absolute path
EVAL_SUPPORT_SIMILARITY_CACHE="auto"       # auto | relative path in pools | absolute path
EVAL_SUPPORT_SIMILARITY_SIZE=32
EVAL_OEM_SPLIT="val"                       # val | test
EVAL_OEM_VAL_JSON="val.json"
EVAL_OEM_TEST_JSON="test.json"
VAIHINGEN_MERGE_CLASSES=(6)           # None: 6 classes | 6: merge Clutter into Imp.Surfaces

timestamp() {
  date +%Y%m%d_%H%M%S
}

datapath_for_benchmark() {
  local benchmark="$1"
  case "$benchmark" in
    oem)
      echo "$OEM_DATAPATH"
      ;;
    vaihingen)
      echo "/scratch/matheuspimenta"
      ;;
    *)
      echo "${BASE_DATAPATH}/${benchmark}"
      ;;
  esac
}

max_way_for_benchmark() {
  local benchmark="$1"
  local bgclass="$2"
  local merge_class="${3:-None}"
  case "$benchmark" in
    pascal|coco)
      echo 5
      ;;
    vaihingen)
      local max_way=6
      if [[ "$merge_class" != "None" ]]; then
        max_way=$((max_way - 1))
      fi
      if [[ "$bgclass" -gt 0 ]]; then
        max_way=$((max_way - 1))
      fi
      echo "$max_way"
      ;;
    chesapeake)
      if [[ "$bgclass" -eq 0 ]]; then
        echo 6
      else
        echo 5
      fi
      ;;
    oem)
      if [[ "$FOLD" -eq 0 ]]; then
        echo 4
      else
        echo 7
      fi
      ;;
    *)
      echo 1
      ;;
  esac
}

run_train() {
  local benchmark="$1"
  local datapath
  datapath="$(datapath_for_benchmark "$benchmark")"

  local run_stamp
  run_stamp="$(timestamp)"
  local logpath
  logpath="experiments/${benchmark}/${METHOD}_shot${TRAIN_SHOT}_bgclass${TRAIN_BGCLASS}_${run_stamp}"
  mkdir -p "$logpath"

  local logfile="${logpath}/train.log"
  local cmd=(
    python main.py
    --datapath "$datapath"
    --benchmark "$benchmark"
    --method "$METHOD"
    --logpath "$logpath"
    --way 1
    --shot "$TRAIN_SHOT"
    --bgclass "$TRAIN_BGCLASS"
    --bsz "$TRAIN_BSZ"
    --fold "$FOLD"
    --backbone "$BACKBONE"
    --niter "$TRAIN_NITER"
  )

  if [[ "$USE_BGD" == "true" ]]; then
    cmd+=(--bgd)
  fi
  if [[ "$USE_NOWANDB" == "true" ]]; then
    cmd+=(--nowandb)
  fi

  echo "[train] benchmark=$benchmark method=$METHOD way=1 shot=$TRAIN_SHOT bgclass=$TRAIN_BGCLASS"
  CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" > "$logfile" 2>&1
  echo "[train] finalizado: $logfile"
}

run_eval_matrix() {
  local run_stamp
  run_stamp="$(timestamp)"

  resolve_eval_ckptpath() {
    local method="$1"
    if [[ -n "$CKPTPATH" ]]; then
      echo "${CKPTPATH//\{method\}/$method}"
    else
      echo "logs/pascal/fold${FOLD}/${BACKBONE}/${method}"
    fi
  }

  for method in "${EVAL_METHODS[@]}"; do
    local ckptpath_method
    ckptpath_method="$(resolve_eval_ckptpath "$method")"
    if [[ ! -e "$ckptpath_method" ]]; then
      echo "[eval] skip method=$method (checkpoint nao encontrado: $ckptpath_method)"
      continue
    fi

    for benchmark in "${EVAL_BENCHMARKS[@]}"; do
      local datapath
      datapath="$(datapath_for_benchmark "$benchmark")"

      # For vaihingen, also loop over merge_class configurations
      local merge_classes_to_test
      if [[ "$benchmark" == "vaihingen" ]]; then
        merge_classes_to_test=("${VAIHINGEN_MERGE_CLASSES[@]}")
      else
        merge_classes_to_test=("None")
      fi

      for merge_class in "${merge_classes_to_test[@]}"; do
        for bgclass in "${EVAL_BGCLASSES[@]}"; do
          local max_way
          max_way="$(max_way_for_benchmark "$benchmark" "$bgclass" "$merge_class")"

          for strategy in "${EVAL_SUPPORT_STRATEGIES[@]}"; do
            for way in "${EVAL_WAYS[@]}"; do
              local effective_way
              effective_way="$way"
              if (( way > max_way )); then
                effective_way="$max_way"
                echo "[eval] adjust benchmark=$benchmark method=$method strategy=$strategy bgclass=$bgclass way=$way -> way=$effective_way (max_way=$max_way)"
              fi

              for shot in "${EVAL_SHOTS[@]}"; do
                local merge_suffix=""
                if [[ "$benchmark" == "vaihingen" && "$merge_class" != "None" ]]; then
                  merge_suffix="_merge${merge_class}"
                fi

                local outdir
                outdir="experiments/eval/${benchmark}/${method}/${strategy}/way${effective_way}_shot${shot}_bgclass${bgclass}${merge_suffix}_${run_stamp}"
                mkdir -p "$outdir"
                local logfile="${outdir}/eval.log"

                local cmd=(
                  python main.py
                  --datapath "$datapath"
                  --benchmark "$benchmark"
                  --method "$method"
                  --logpath "$outdir"
                  --ckptpath "$ckptpath_method"
                  --way "$effective_way"
                  --shot "$shot"
                  --bgclass "$bgclass"
                  --bsz "$EVAL_BSZ"
                  --fold "$FOLD"
                  --backbone "$BACKBONE"
                  --support_strategy "$strategy"
                  --support_area_cache "$EVAL_SUPPORT_AREA_CACHE"
                  --support_similarity_cache "$EVAL_SUPPORT_SIMILARITY_CACHE"
                  --support_similarity_size "$EVAL_SUPPORT_SIMILARITY_SIZE"
                  --eval
                )

                if [[ "$benchmark" == "oem" ]]; then
                  cmd+=(--oem_eval_split "$EVAL_OEM_SPLIT")
                  cmd+=(--oem_val_json "$EVAL_OEM_VAL_JSON")
                  cmd+=(--oem_test_json "$EVAL_OEM_TEST_JSON")
                fi

                # Add merge_class parameter for vaihingen
                if [[ "$benchmark" == "vaihingen" && "$merge_class" != "None" ]]; then
                  cmd+=(--merge_class "$merge_class")
                fi

                if [[ "$USE_BGD" == "true" ]]; then
                  cmd+=(--bgd)
                fi
                if [[ "$USE_NOWANDB" == "true" ]]; then
                  cmd+=(--nowandb)
                fi

                local merge_info=""
                if [[ "$benchmark" == "vaihingen" ]]; then
                  if [[ "$merge_class" == "None" ]]; then
                    merge_info=" (6-class)"
                  else
                    merge_info=" (merge_class=$merge_class)"
                  fi
                fi
                echo "[eval] benchmark=$benchmark method=$method strategy=$strategy way=$effective_way shot=$shot bgclass=$bgclass${merge_info}"
                CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" > "$logfile" 2>&1
                echo "[eval] finalizado: $logfile"
              done
            done
          done
        done
      done
    done
  done
}

case "$MODE" in
  train)
    run_train "$TRAIN_BENCHMARK"
    ;;
  eval)
    run_eval_matrix
    ;;
  *)
    echo "MODE invalido: $MODE (use train ou eval)"
    exit 1
    ;;
esac