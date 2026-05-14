r""" Vaihingen few-shot classification and segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

TOTAL_TILES = 819

class DatasetVAIHINGEN (Dataset):
    """
    FS-CS Vaihingen dataset of which split follows the standard FS-S dataset
    """
    def __init__(self, datapath, fold, transform, split, way, shot, bgclass, bgd, rdn_sup,
                 merge_class=None):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.nfolds = 1
        # Base: 6 classes (Imp. Surf, Buildings, Low Veg, Trees, Cars, Clutter)
        # If merge_class is set, merge that class into class 1 (Imp. Surfaces)
        self.merge_class = merge_class
        self.nclass = 6
        if merge_class is not None and merge_class != 1:
            self.nclass -= 1
        self.nclass -= (1 if bgclass != 0 else 0)
        self.benchmark = 'vaihingen'

        self.fold = fold
        self.way = way
        self.shot = shot
        self.bgclass = bgclass

        self.img_path = os.path.join(datapath, 'Vaihingen/tiles/images/')
        self.ann_path = os.path.join(datapath, 'Vaihingen/tiles/masks/')
        self.pool_path = '/home/matheuspimenta/Jobs/SR/ifsl/utils/vaihingen/pool_files'
        self.transform = transform

        self.bgd = bgd
        self.mask_palette = {
            1: (255, 255, 255),  # Impervious surfaces (white)
            2: (0, 0, 255),  # Buildings (blue)
            3: (0, 255, 255),  # Low vegetation (cyan)
            4: (0, 255, 0),  # Trees (green)
            5: (255, 255, 0),  # Cars (yellow)
            6: (255, 0, 0),  # Clutter (red)
            0: (0, 0, 0)  # Undefined (black)
        }

        # Build original class IDs and remapped IDs (analogous to Chesapeake)
        self.class_ids_orig = [1, 2, 3, 4, 5, 6]
        if merge_class is not None and merge_class != 1:
            self.class_ids_orig.remove(merge_class)
        if bgclass > 0:
            self.class_ids_orig.remove(bgclass)
        self.class_ids = [self.remap_class_id(class_id) for class_id in self.class_ids_orig]
        
        if self.way > len(self.class_ids):
            raise ValueError(f'way={self.way} is larger than available classes={len(self.class_ids)} for split={self.split}')
        
        self.query_pool, self.support_pool = self.build_pools() 
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()
        self.rdn_sup = rdn_sup

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        query_name, support_names, _support_classes = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_class_presence = [s_c in torch.unique(query_cmask) for s_c in _support_classes]  # needed - 1
        rename_class = lambda x: _support_classes.index(x) + 1 if x in _support_classes else 0 # funcao que retorna indice+1 
                                                                                               # se tiver a classe

        query_img = self.transform(query_img)
        query_mask, query_ignore_idx = self.get_query_mask(query_img, query_cmask, rename_class)
        support_imgs = torch.stack([torch.stack([self.transform(support_img) for support_img in support_imgs_c]) for support_imgs_c in support_imgs])
        support_masks, support_ignore_idxs = self.get_support_masks(support_imgs, _support_classes, support_cmasks, rename_class)

        _support_classes = torch.tensor(_support_classes)
        query_class_presence = torch.tensor(query_class_presence)

        assert query_class_presence.int().sum() == (len(torch.unique(query_mask)) - (1 if (self.bgd and 0 in torch.unique(query_mask)) else 0))

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'support_classes': _support_classes,
                 'query_class_presence': query_class_presence}

        return batch

    def get_query_mask(self, query_img, query_cmask, rename_class):
        if self.split == 'trn':  # resize during training and retain orignal sizes during validation
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.generate_query_episodic_mask(query_cmask.float(), rename_class)
        return query_mask, query_ignore_idx

    def get_support_masks(self, support_imgs, _support_classes, support_cmasks, rename_class):
        support_masks = []
        support_ignore_idxs = []
        for class_id, scmask_c in zip(_support_classes, support_cmasks):  # ways
            support_masks_c = []
            support_ignore_idxs_c = []
            for scmask in scmask_c:  # shots
                scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
                support_mask, support_ignore_idx = self.generate_support_episodic_mask(scmask, class_id, rename_class)
                assert len(torch.unique(support_mask)) <= 2, f'{len(torch.unique(support_mask))} labels in support'
                support_masks_c.append(support_mask)
                support_ignore_idxs_c.append(support_ignore_idx)
            support_masks.append(torch.stack(support_masks_c))
            support_ignore_idxs.append(torch.stack(support_ignore_idxs_c))
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)
        return support_masks, support_ignore_idxs

    def generate_query_episodic_mask(self, mask, rename_class):
        # mask = mask.clone()
        mask_renamed = torch.zeros_like(mask).to(mask.device).type(mask.dtype)
        boundary = (mask / 255).floor()

        classes = torch.unique(mask)
        for c in classes:
            mask_renamed[mask == c] = 0 if c in [0, 255] else rename_class(c)

        return mask_renamed, boundary

    def generate_support_episodic_mask(self, mask, class_id, rename_class):
        mask = mask.clone()
        boundary = (mask / 255).floor()
        mask[mask != class_id] = 0
        mask[mask == class_id] = rename_class(class_id)

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img  = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs  = [[self.read_img(name)  for name in support_names_c] for support_names_c in support_names]
        support_masks = [[self.read_mask(name) for name in support_names_c] for support_names_c in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name))
    
    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(self.convert_mask(np.array(Image.open(os.path.join(self.ann_path, img_name)))))
        return mask

    def convert_mask(self, np_mask):
        # Class
        height, width, _ = np_mask.shape
        new_mask = np.zeros((height, width), dtype=np.uint8)

        # Cria um array para mapear cores para índices
        color_to_index = np.zeros((256, 256, 256), dtype=np.uint8)

        for index, color in self.mask_palette.items():
            color_to_index[color] = index

        # Reshape da máscara original para facilitar o mapeamento
        flat_mask = np_mask.reshape(-1, 3)

        # Mapeamento das cores para índices
        flat_new_mask = color_to_index[flat_mask[:, 0], flat_mask[:, 1], flat_mask[:, 2]]

        # Reshape da máscara de volta para a forma original
        new_mask = flat_new_mask.reshape(height, width)
        
        # Handle merge_class: merge into class 1
        if self.merge_class is not None and self.merge_class != 1:
            new_mask[new_mask == self.merge_class] = 1
        
        # Handle bgclass: shift down classes above bgclass
        if self.bgclass > 0:
            new_mask[new_mask == self.bgclass] = 0
            new_mask[new_mask > self.bgclass] -= 1

        return new_mask

    def remap_class_id(self, class_id):
        """Convert original class ID to remapped ID accounting for merge_class and bgclass."""
        remapped = class_id
        
        # Account for merge_class: classes above merge_class shift down by 1
        if self.merge_class is not None and self.merge_class != 1 and class_id > self.merge_class:
            remapped -= 1
        
        # Account for bgclass: classes above bgclass shift down by 1
        if self.bgclass > 0 and remapped > self.bgclass:
            remapped -= 1
        
        return remapped

    def sample_episode(self, idx):
        # Fix (q, s) pair for all queries across different batch sizes for reproducibility
        if self.split == 'val':
            np.random.seed(idx)

        idx %= len(self.img_metadata)
        query_name, query_class = self.img_metadata[idx]

        # 3-way 2-shot support_names: [[c1_1, c1_2], [c2_1, c2_2], [c3_1, c3_2]]
        support_names = []

        if self.way == 1:
            support_classes = [query_class]
        else:
            p = np.ones([len(self.class_ids)]) / 2. / float(len(self.class_ids) - 1)
            p[self.class_ids.index(query_class)] = 1 / 2.
            support_classes = np.random.choice(self.class_ids, self.way, p=p, replace=False).tolist()

        def sample_from_pool(pool, query_name, shot):
            candidates = [name for name in pool if name != query_name]
            if not candidates:
                candidates = list(pool)
            if not candidates:
                raise RuntimeError('Empty support candidates while sampling episode')
            replace = len(candidates) < shot
            return np.random.choice(candidates, shot, replace=replace).tolist()

        if self.rdn_sup:
            # For random, use all images from support_pool[21] (all classes combined)
            for _ in support_classes:
                support_names.append(sample_from_pool(self.support_pool[21], query_name, self.shot))
        else:
            # For class-specific pools, use remapped class IDs to read original pool files
            for sc in support_classes:
                pool_key = self.class_ids_orig[self.class_ids.index(sc)]
                if len(self.support_pool[pool_key]) == 0:
                    raise RuntimeError(f'Empty support pool for original class {pool_key} (remapped {sc}) in {self.pool_path}')
                support_names.append(sample_from_pool(self.support_pool[pool_key], query_name, self.shot))

        return query_name, support_names, support_classes

    # Read query and supports files
    def build_pools(self):
        query_pool = []
        query_file = os.path.join(self.pool_path, f'querys.txt')
        with open(query_file, 'r') as f:
            query_pool = f.read().split('\n')[:-1]

        # Initialize pools for original class IDs (1-6)
        support_pool = {i: [] for i in range(1, 7)}
        support_pool[21] = []  # combined pool for random strategy
        
        for i in range(1, 7):
            support_file = os.path.join(self.pool_path, f'{i}.txt')
            if os.path.exists(support_file):
                with open(support_file, 'r') as f:
                    support_pool[i] = f.read().split('\n')[:-1]
                    support_pool[21] += support_pool[i]
            else:
                # Skip if file doesn't exist (might happen if class was merged)
                support_pool[i] = []

        return query_pool, support_pool
    
    
    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        query_set = set(self.query_pool)

        for idx, class_id in enumerate(self.class_ids):
            # Map remapped class ID back to original class ID for pool reading
            original_class_id = self.class_ids_orig[idx]
            
            if self.split == 'trn':
                candidates = list(self.support_pool[original_class_id])
            else:
                candidates = [name for name in self.support_pool[original_class_id] if name in query_set]
                if len(candidates) == 0:
                    candidates = list(self.support_pool[original_class_id])

            if len(candidates) == 0:
                raise RuntimeError(f'No metadata found for remapped class {class_id} (original {original_class_id}) in split {self.split}')

            img_metadata_classwise[class_id] = candidates

        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for class_id in self.class_ids:
            img_metadata += [[img_name, class_id] for img_name in self.img_metadata_classwise[class_id]]

        if len(img_metadata) == 0:
            raise RuntimeError(f'No image metadata available for split {self.split}')

        print(f'Total {self.split} images are : {len(img_metadata):,}')
        return img_metadata

