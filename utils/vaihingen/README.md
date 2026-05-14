# Vaihingen Pool Files & Dataset Setup

## Directory Structure

```
/home/matheuspimenta/Jobs/SR/ifsl/utils/vaihingen/
├── pool_files/                    # Support/query pools for each class
│   ├── 1.txt                      # Class 1: Impervious Surfaces (683 images, 80.7%)
│   ├── 2.txt                      # Class 2: Buildings (703 images, 83.1%)
│   ├── 3.txt                      # Class 3: Low Vegetation (628 images, 74.2%)
│   ├── 4.txt                      # Class 4: Trees (594 images, 70.2%)
│   ├── 5.txt                      # Class 5: Cars (6 images, 0.7%)
│   ├── 6.txt                      # Class 6: Clutter (18 images, 2.1%)
│   └── querys.txt                 # All 846 images for query set
├── analyze_mask_distribution.py   # Utility for analyzing mask distributions
├── generate_pool_files.py         # Utility to generate pool files (10% threshold)
└── class_distribution.json        # Full class statistics
```

## Pool Generation

**Threshold**: 10% class presence per image
- Classes must occupy ≥10% of image pixels to be included in that image's support pool
- This filters out noise while capturing meaningful representation

**Generation command**:
```bash
python /home/matheuspimenta/Jobs/SR/ifsl/utils/vaihingen/generate_pool_files.py
```

## Dataset Parameters

### merge_class (NEW)
- **Default**: `None` (keep all 6 classes)
- **Usage**: `merge_class=6` merges Clutter (class 6) into Impervious Surfaces (class 1)
- **Purpose**: Test whether the sparse Clutter class (2.1% of images) benefits from merging with the dominant class
- **Effect**: 
  - When set to 6: `nclass=5` instead of 6
  - Clutter pixels are converted to class 1 during mask loading
  - Class remapping: original [1,2,3,4,5,6] → remapped [1,2,3,4,5]

### bgclass
- **Default**: `0` (no background class)
- **Standard behavior**: Use `bgclass=k` to set class k as episodic background (becomes 0, others shift down)
- **Example**: `bgclass=3, merge_class=6` → classes [1,2,3,4,5] but 3 is background → active classes [1,2,4,5] remapped to [1,2,3,4]

## Class Mapping

| Original ID | Class Name | RGB | Pool Size | merge_class=None | merge_class=6 |
|------------|------------|-----|-----------|-----------------|---------------|
| 1 | Impervious Surfaces | (255,255,255) | 683 | 1 | 1 (merged with 6) |
| 2 | Buildings | (0,0,255) | 703 | 2 | 2 |
| 3 | Low Vegetation | (0,255,255) | 628 | 3 | 3 |
| 4 | Trees | (0,255,0) | 594 | 4 | 4 |
| 5 | Cars | (255,255,0) | 6 | 5 | 5 |
| 6 | Clutter | (255,0,0) | 18 | 6 | (merged to 1) |

## Usage Examples

```python
from data.vaihingen import DatasetVAIHINGEN
from torchvision.transforms import Compose, ToTensor

transform = Compose([ToTensor()])

# 6-class Vaihingen (baseline)
ds_baseline = DatasetVAIHINGEN(
    datapath='/scratch/matheuspimenta',
    fold=0,
    transform=transform,
    split='val',
    way=3,
    shot=1,
    bgclass=0,
    bgd=False,
    rdn_sup=False,
    merge_class=None
)

# 5-class with Clutter merged into Impervious Surfaces
ds_merged = DatasetVAIHINGEN(
    datapath='/scratch/matheuspimenta',
    fold=0,
    transform=transform,
    split='val',
    way=3,
    shot=1,
    bgclass=0,
    bgd=False,
    rdn_sup=False,
    merge_class=6
)
```

## Notes

1. **Class imbalance**: Yellow (cars) and especially Red (clutter) are underrepresented. The 10% threshold helps but these remain sparse.
2. **Pool path**: Dataset reads from `/home/matheuspimenta/Jobs/SR/ifsl/utils/vaihingen/pool_files/`
3. **Supported strategies** (with pool files):
   - `rdn_sup=True`: Random sampling from combined pool (all classes)
   - `rdn_sup=False`: Class-specific sampling (each class from its own pool)
4. **max_area** and **similarity** strategies: Not yet implemented for Vaihingen; could be added similar to Chesapeake if needed
