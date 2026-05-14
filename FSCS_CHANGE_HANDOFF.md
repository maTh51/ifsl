# FS-CS Change Handoff

This document records the changes currently present in the `ifsl` workspace so they can be recreated on another machine without having to rediscover the design from scratch.

The work is mostly concentrated in `fs-cs/` plus generated pool artifacts under `utils/`. The main goals were:

1. Add stronger dataset-specific support handling for Chesapeake and Vaihingen.
2. Introduce a merged-class option for Vaihingen.
3. Add OEM support as a new dataset path, including debug notebooks and optional large-image inference.
4. Make evaluation and checkpoint loading more flexible.
5. Generate and store the support/query pool files used by the episodic samplers.

## What Changed

### 1) Shared runtime and CLI plumbing

Files:

- `fs-cs/main.py`
- `fs-cs/data/dataset.py`
- `fs-cs/common/callbacks.py`
- `fs-cs/common/evaluation.py`
- `fs-cs/model/ifsl.py`

Key changes:

- Added new CLI options for dataset support strategies and OEM settings.
- Added `--merge_class` for Vaihingen.
- Added `--support_strategy`, `--support_area_cache`, `--support_similarity_cache`, and `--support_similarity_size`.
- Added OEM options: `--oem_train_list`, `--oem_val_json`, `--oem_test_json`, `--oem_crop_size`, `--oem_sw_enable`, `--oem_sw_tile`, `--oem_sw_stride`.
- `fs-cs/data/dataset.py` now forwards the new dataset arguments only to the benchmarks that use them.
- `fs-cs/common/callbacks.py` can now load a checkpoint from an explicit file or from a directory that contains the best checkpoint.
- `fs-cs/common/evaluation.py` now uses a confusion-matrix-based F1 implementation and supports binary or boolean ignore masks.
- `fs-cs/model/ifsl.py` now has optional sliding-window inference for OEM when the raw query image is available.

Important behavior:

- `compute_cls_er()` still returns an accuracy-like value for backward compatibility, even though the name is historical.
- `compute_cls_error_rate()` returns the actual error rate as `100 - accuracy`.
- When evaluation is run with `--ckptpath`, the code uses that path directly instead of auto-resolving the best model under the log directory.

### 2) Chesapeake dataset updates

File:

- `fs-cs/data/chesapeake.py`

Key changes:

- The dataset now reads support/query pool files from `utils/chesapeake/` instead of relying on the previous root-level pool files.
- Added support sampling strategies:
  - `random`
  - `max_area`
  - `similarity`
- Added optional cache loading/building for support ranking:
  - class representativity cache: `chesapeake_class_representativity.json`
  - similarity feature cache: `chesapeake_similarity_index.npz`
- Added class remapping logic for `bgclass` so the episodic labels remain contiguous after one class is reserved as background.
- Query and support metadata now use remapped class IDs consistently.

Notes:

- `max_area` ranks candidates by per-class pixel representativity in the support pool.
- `similarity` ranks candidates by a simple normalized RGB feature similarity.
- `random` remains the fallback and the default.
- The caches are generated automatically the first time they are needed, so they are not required to be stored in version control for the handoff.

### 3) Vaihingen dataset updates

Files:

- `fs-cs/data/vaihingen.py`
- `utils/vaihingen/generate_pool_files.py`
- `utils/vaihingen/README.md`
- `utils/vaihingen/class_distribution.json`
- `utils/vaihingen/class_distribution.csv`
- `utils/vaihingen/pool_files/*.txt`

Key changes:

- Added `merge_class` support to `DatasetVAIHINGEN`.
- `merge_class=6` merges Clutter into Impervious Surfaces.
- The dataset now builds class IDs from the merged class set and then applies `bgclass` remapping if needed.
- Support/query pools now come from `utils/vaihingen/pool_files/`.
- The generated support pools are based on a 10% minimum class-presence threshold per image.

Behavioral details:

- Original class IDs are `1..6`.
- If `merge_class=6`, the effective number of classes becomes 5 before background handling.
- If `bgclass > 0`, the selected background class is removed from the episodic class list and remaining labels shift down to stay contiguous.

Pool generation:

- `utils/vaihingen/generate_pool_files.py` scans mask PNGs and writes:
  - `pool_files/1.txt` through `pool_files/6.txt`
  - `pool_files/querys.txt`
- `utils/vaihingen/class_distribution.json` and `utils/vaihingen/class_distribution.csv` are generated analysis outputs summarizing class coverage.

### 4) OEM dataset addition

File:

- `fs-cs/data/oem.py`

Key changes:

- Added a new `DatasetOEM` loader.
- Uses official train/val/test split definitions.
- Supports episodic sampling for training and evaluation.
- Training can apply random crops of size `oem_crop_size`.
- Validation/test can preserve the raw query image in `query_img_raw` for optional sliding-window inference.
- Query masks can be missing for some samples; such episodes are tracked with `has_query_mask` so the evaluation code can skip unlabeled samples.

Important details:

- OEM support strategy is also wired through `random`, `max_area`, and `similarity`.
- The similarity and max-area caches follow the same pattern as Chesapeake, but are namespaced per split.
- Large-image inference is only activated when `--oem_sw_enable` is set.

### 5) Eval orchestration and shell workflow

Files:

- `fs-cs/run.sh`
- `fs-cs/experiments/` and `fs-cs/logs/` contain runtime outputs, not source logic.

Key changes:

- `run.sh` now orchestrates a matrix of methods, benchmarks, support strategies, ways, shots, background classes, and, for Vaihingen, `merge_class` values.
- It passes the new dataset arguments to `main.py`.
- It constructs benchmark-specific log directories and evaluation output directories.
- It resolves a checkpoint path from either a file or a directory.

Current intent of the shell workflow:

- Train one benchmark at a time.
- Evaluate multiple methods and support strategies in a sweep.
- Keep the configuration surface in one place instead of spreading it across ad hoc commands.

## Support Strategy Model

The support selection logic now follows a shared conceptual model across the project:

- `random`: sample supports uniformly from the pool.
- `max_area`: rank support candidates by how much of the target class they contain.
- `similarity`: rank support candidates by a precomputed or on-demand similarity feature.

In the current implementation:

- Chesapeake and OEM actively use all three strategies.
- Vaihingen primarily uses the regenerated class pools and the `merge_class` path; the runtime sweep still exposes strategy names, but the loader itself is not the same as Chesapeake/OEM.

## Generated Artifacts To Recreate

Include these in the handoff because they are part of the working setup:

### Chesapeake pools and analysis artifacts

- `utils/chesapeake/1.txt`
- `utils/chesapeake/2.txt`
- `utils/chesapeake/3.txt`
- `utils/chesapeake/4.txt`
- `utils/chesapeake/5.txt`
- `utils/chesapeake/6.txt`
- `utils/chesapeake/querys.txt`
- `utils/chesapeake/chesapeake_class_representativity.json`
- `utils/chesapeake/chesapeake_similarity_index.npz`

The text files are the important source artifact here. The JSON and NPZ files are generated caches; copy them if you want the exact same runtime state, but they can also be regenerated on demand.

### Vaihingen pools and analysis artifacts

- `utils/vaihingen/pool_files/1.txt`
- `utils/vaihingen/pool_files/2.txt`
- `utils/vaihingen/pool_files/3.txt`
- `utils/vaihingen/pool_files/4.txt`
- `utils/vaihingen/pool_files/5.txt`
- `utils/vaihingen/pool_files/6.txt`
- `utils/vaihingen/pool_files/querys.txt`
- `utils/vaihingen/class_distribution.json`
- `utils/vaihingen/class_distribution.csv`
- `utils/vaihingen/README.md`
- `utils/vaihingen/generate_pool_files.py`
- `utils/vaihingen/analyze_mask_distribution.py`

### Debug-only notebooks

These are useful for inspection but should be treated as debug-only, not production dependencies:

- `fs-cs/analysis.ipynb`
- `fs-cs/oem_val_episode_debug.ipynb`
- `fs-cs/oem_val_visual_debug.ipynb`

## Rebuild Steps On Another Machine

1. Clone the repo and make sure the same Python environment and dataset dependencies are installed.
2. Update the hardcoded paths in the datasets and shell script to match the new machine.
3. Recreate the Vaihingen pool files by running `utils/vaihingen/generate_pool_files.py` against the local mask directory.
4. Copy or regenerate the Chesapeake text pools under `utils/chesapeake/`.
5. Let Chesapeake build its JSON and NPZ caches on first use, or copy them if you want a fully identical runtime state.
6. Run the matrix or one-off commands through `fs-cs/run.sh`.

Example commands:

```bash
python fs-cs/main.py --help
bash fs-cs/run.sh
python utils/vaihingen/generate_pool_files.py --help
python utils/vaihingen/analyze_mask_distribution.py --help
```

## Porting Notes

- `DatasetVAIHINGEN` currently reads pools from `/home/matheuspimenta/Jobs/SR/ifsl/utils/vaihingen/pool_files`.
- `DatasetCHESAPEAKE` currently reads pools from `/home/matheuspimenta/Jobs/SR/ifsl/utils/chesapeake`.
- `DatasetOEM` defaults to a pool directory under the OEM datapath.
- `run.sh` assumes machine-specific dataset roots such as `/scratch/dataset` and `/scratch/matheuspimenta/oem`; those must be edited before reuse.
- Chesapeake and Vaihingen checkpoints are written under `logs/pascal/fold...` by the current callback logic, even though the benchmark is not Pascal. That path choice is part of the current behavior and should be preserved if you want identical log layout.

## Suggested Handoff Summary

If you need to rebuild this state elsewhere, the minimum source files to copy are:

- `fs-cs/main.py`
- `fs-cs/data/dataset.py`
- `fs-cs/data/chesapeake.py`
- `fs-cs/data/vaihingen.py`
- `fs-cs/data/oem.py`
- `fs-cs/common/callbacks.py`
- `fs-cs/common/evaluation.py`
- `fs-cs/model/ifsl.py`
- `fs-cs/run.sh`
- `utils/chesapeake/*.txt`
- `utils/vaihingen/pool_files/*.txt`
- `utils/vaihingen/generate_pool_files.py`
- `utils/vaihingen/analyze_mask_distribution.py`
- `utils/vaihingen/README.md`

The generated caches can be copied afterward if you want byte-for-byte reproducibility, but they are not required for understanding the implementation.