r""" Chesapeake few-shot classification and segmentation dataset """
import os
import json

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import random
import time

TOTAL_TILES = 819

class DatasetCHESAPEAKE (Dataset):
    """
    FS-CS Chesapeake dataset of which split follows the standard FS-S dataset
    """
    def __init__(self, datapath, fold, transform, split, way, shot, bgclass, bgd, rdn_sup,
                 support_strategy='random', support_area_cache='auto',
                 support_similarity_cache='auto', support_similarity_size=32, use_infrared=False):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.nfolds = 1
        self.nclass = 6 - (1 if bgclass != 0 else -0)
        self.benchmark = 'chesapeake'

        self.fold = fold
        self.way = way
        self.shot = shot
        self.bgclass = bgclass

        self.img_path = os.path.join(datapath)
        self.ann_path = os.path.join(datapath)
        # self.pool_path = os.path.join(datapath, 'chesapeake_400/pools/')
        self.pool_path = '/home/matheuspimenta/Jobs/SR/ifsl/utils/chesapeake'
        self.transform = transform

        self.bgd = bgd
        # self.mask_palette = {
        #     0:  0,    # não existe no dataset
        #     1:  1,    # agua
        #     2:  2,    # floresta
        #     3:  3,    # campo
        #     4:  4,    # terra estéril
        #     5:  5,    # impermeável (outro)
        #     6:  6,    # impermeável (estrada)
        #     15: 0,   # sem dados
        # }

        self.class_ids_orig = [1, 2, 3, 4, 5, 6]
        if self.bgclass > 0:
            self.class_ids_orig.remove(self.bgclass)
        self.class_ids = [self.remap_class_id(class_id) for class_id in self.class_ids_orig]
        if self.way > len(self.class_ids):
            raise ValueError(f'way={self.way} is larger than available classes={len(self.class_ids)} for split={self.split}')
        self.query_pool, self.support_pool = self.build_pools() 
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()
        self.rdn_sup = rdn_sup

        self.support_strategy = support_strategy
        self.use_infrared = bool(use_infrared)
        valid_strategies = {'random', 'max_area', 'similarity', 'similarity_global'}
        if self.support_strategy not in valid_strategies:
            raise ValueError(f'Invalid support strategy={self.support_strategy}. Choices: {sorted(valid_strategies)}')

        self.class_representativity = None
        self.similarity_entries = None
        self.similarity_features = None
        self.similarity_index = None
        self.query_feature_cache = {}
        self.similarity_size = int(support_similarity_size)

        if self.split == 'val' and self.support_strategy == 'max_area':
            area_cache_path = self.resolve_cache_path(support_area_cache, 'chesapeake_class_representativity.json')
            self.class_representativity = self.load_or_build_class_representativity(area_cache_path)

        if self.split == 'val' and self.support_strategy in ('similarity', 'similarity_global'):
            sim_cache_path = self.resolve_cache_path(support_similarity_cache, 'chesapeake_similarity_index.npz')
            (self.similarity_entries,
             self.similarity_features,
             self.similarity_index) = self.load_or_build_similarity_index(sim_cache_path)

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

        if query_class_presence.int().sum() != (len(torch.unique(query_mask)) - (1 if (self.bgd and 0 in torch.unique(query_mask)) else 0)):
            print(query_class_presence.int().sum() != (len(torch.unique(query_mask)) - (1 if (self.bgd and 0 in torch.unique(query_mask)) else 0)))
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
        rgb = self.read_img_array(img_name)
        return Image.fromarray(rgb, 'RGB')

    def read_img_array(self, img_name):
        r"""Return RGB image as numpy uint8 array."""
        path, dims = img_name.split(";")
        dims = dims.split(",")
        window = Window(int(dims[0]), int(dims[1]), 400, 400)
        with rio.open(os.path.join(self.img_path, path.replace( "lc.tif", "naip-new.tif"))) as src:
            # If requested and available, replace red channel with infrared (band 4)
            if self.use_infrared and getattr(src, 'count', 0) >= 4:
                infrared = src.read(4, window=window)
                green = src.read(2, window=window)
                blue = src.read(3, window=window)
                rgb = np.dstack((infrared, green, blue))
            else:
                red = src.read(1, window=window)
                green = src.read(2, window=window)
                blue = src.read(3, window=window)
                rgb = np.dstack((red, green, blue))
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb
    
    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        # Image.open(os.path.join(self.ann_path, img_name))
        path, dims = img_name.split(";")
        dims = dims.split(",")
        window = Window(int(dims[0]), int(dims[1]), 400, 400)
        rast_mask = rio.open(os.path.join(self.ann_path, path)).read(1, window=window)
        # mask = torch.tensor(self.convert_mask(np.moveaxis(rast_mask, 0, -1)))
        if self.bgclass != 0:
            rast_mask = self.convert_mask(rast_mask)
        return torch.tensor(rast_mask)

    def convert_mask(self, np_mask):
        transformed_mask = np.copy(np_mask)
        if self.bgclass > 0:
            transformed_mask[transformed_mask == self.bgclass] = 0
            transformed_mask[transformed_mask > self.bgclass] -= 1
        
        return transformed_mask

    def remap_class_id(self, class_id):
        if self.bgclass > 0 and class_id > self.bgclass:
            return class_id - 1
        return class_id

    def sample_episode(self, idx):
        # Fix (q, s) pair for all queries across different batch sizes for reproducibility
        if self.split == 'val':
            np.random.seed(idx)

        idx %= len(self.img_metadata)
        query_name, query_class = self.img_metadata[idx]
        # black_list = ['top_mosaic_09cm_area30/part9.png','top_mosaic_09cm_area15/part2.png','top_mosaic_09cm_area15/part13.png']
        # while query_name in black_list:
        #     idx = idx + 1
        #     idx %= len(self.query_pool)
        #     query_name = self.query_pool[idx]


        # 3-way 2-shot support_names: [[c1_1, c1_2], [c2_1, c2_2], [c3_1, c3_2]]
        support_names = []
        global_similarity_chunks = None

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

        def select_support(pool, query_name, shot, class_id=None):
            if self.split != 'val' or self.support_strategy == 'random':
                return sample_from_pool(pool, query_name, shot)
            if self.support_strategy == 'max_area' and class_id is not None:
                return self.sample_max_area_support(pool, query_name, shot, class_id)
            if self.support_strategy == 'similarity':
                return self.sample_similarity_support(pool, query_name, shot)
            return sample_from_pool(pool, query_name, shot)

        if self.split == 'val' and self.support_strategy == 'similarity_global':
            total_needed = int(self.way) * int(self.shot)
            global_list = self.sample_similarity_global(query_name, total_needed)
            global_similarity_chunks = [global_list[i * int(self.shot):(i + 1) * int(self.shot)] for i in range(int(self.way))]

        if self.rdn_sup:
            for _ in support_classes:
                support_names.append(select_support(self.support_pool[21], query_name, self.shot, class_id=None))
        else:
            if global_similarity_chunks is not None:
                for chunk in global_similarity_chunks:
                    support_names.append(chunk)
            else:
                for sc in support_classes:
                    if len(self.support_pool[sc]) == 0:
                        raise RuntimeError(f'Empty support pool for class {sc} in {self.pool_path}')
                    support_names.append(select_support(self.support_pool[sc], query_name, self.shot, class_id=sc))

        return query_name, support_names, support_classes

    def build_class_ids(self):
        nclass_val = self.nclass // self.nfolds
        # e.g. fold0 val: 1, 2, 3, 4, 5
        class_ids_val = [self.fold * nclass_val + i for i in range(1, nclass_val + 1)]
        # e.g. fold0 trn: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
        class_ids_trn = [x for x in range(1, self.nclass + 1) if x not in class_ids_val]

        assert len(set(class_ids_trn + class_ids_val)) == self.nclass
        assert 0 not in class_ids_val
        assert 0 not in class_ids_trn

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val
    
    # Read query and supports files
    def build_pools(self):
        query_pool = []
        query_file = os.path.join(self.pool_path, f'querys.txt')
        with open(query_file, 'r') as f:
            query_pool = f.read().split('\n')[:-1]

        support_pool = {i: [] for i in self.class_ids}
        support_pool[21] = []
        support_pool[30] = []

        for original_class_id in self.class_ids_orig:
            mapped_class_id = self.remap_class_id(original_class_id)
            support_file = os.path.join(self.pool_path, f'{original_class_id}.txt')
            with open(support_file, 'r') as f:
                support_pool[mapped_class_id] = f.read().split('\n')[:-1]
                support_pool[21] += support_pool[mapped_class_id]
                if support_pool[mapped_class_id]:
                    selected_images = random.sample(support_pool[mapped_class_id], min(20, len(support_pool[mapped_class_id])))
                    support_pool[30] += selected_images

        # for v in support_pool.values():
        #     print(len(v))
        # return query_pool, support_pool
        # return random.sample(support_pool[21], 100), support_pool
        return query_pool, support_pool

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        query_set = set(self.query_pool)

        for class_id in self.class_ids:
            if self.split == 'trn':
                candidates = list(self.support_pool[class_id])
            else:
                candidates = [name for name in self.support_pool[class_id] if name in query_set]
                if len(candidates) == 0:
                    candidates = list(self.support_pool[class_id])

            if len(candidates) == 0:
                raise RuntimeError(f'No metadata found for class {class_id} in split {self.split}')

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

    def resolve_cache_path(self, raw_path, default_name):
        if raw_path in ['', 'auto']:
            return os.path.join(self.pool_path, default_name)
        raw_path = os.path.expanduser(raw_path)
        if os.path.isabs(raw_path):
            return raw_path
        return os.path.join(self.pool_path, raw_path)

    def sample_max_area_support(self, pool, query_name, shot, class_id):
        candidates = [name for name in pool if name != query_name]
        if not candidates:
            candidates = list(pool)
        if not candidates:
            raise RuntimeError('Empty support candidates while sampling max_area support')

        class_scores = self.class_representativity.get(class_id, {}) if self.class_representativity else {}
        ranked = sorted(candidates, key=lambda name: class_scores.get(name, 0.0), reverse=True)
        if len(ranked) >= shot:
            return ranked[:shot]
        if len(ranked) == 1:
            return ranked * shot
        return ranked + np.random.choice(ranked, shot - len(ranked), replace=True).tolist()

    def sample_similarity_support(self, pool, query_name, shot):
        candidates = [name for name in pool if name != query_name]
        if not candidates:
            candidates = list(pool)
        if not candidates:
            raise RuntimeError('Empty support candidates while sampling similarity support')

        query_feat = self.get_query_feature(query_name)
        candidate_pairs = [(name, self.similarity_index[name]) for name in candidates if name in self.similarity_index]
        candidate_indices = [idx for _, idx in candidate_pairs]
        if len(candidate_indices) == 0:
            replace = len(candidates) < shot
            return np.random.choice(candidates, shot, replace=replace).tolist()

        candidate_feats = self.similarity_features[candidate_indices]
        sims = np.dot(candidate_feats, query_feat)
        ranked_pos = np.argsort(-sims)
        ranked_names = [candidate_pairs[i][0] for i in ranked_pos]

        if len(ranked_names) >= shot:
            return ranked_names[:shot]
        if len(ranked_names) == 0:
            replace = len(candidates) < shot
            return np.random.choice(candidates, shot, replace=replace).tolist()
        if len(ranked_names) == 1:
            return ranked_names * shot
        return ranked_names + np.random.choice(ranked_names, shot - len(ranked_names), replace=True).tolist()

    def sample_similarity_global(self, query_name, total):
        candidates = [name for name in self.support_pool[21] if name != query_name]
        if not candidates:
            candidates = list(self.support_pool[21])
        if not candidates:
            raise RuntimeError('Empty support candidates while sampling similarity_global support')

        query_feat = self.get_query_feature(query_name)
        candidate_pairs = [(name, self.similarity_index[name]) for name in candidates if name in self.similarity_index]
        candidate_indices = [idx for _, idx in candidate_pairs]
        if len(candidate_indices) == 0:
            replace = len(candidates) < total
            return np.random.choice(candidates, total, replace=replace).tolist()

        candidate_feats = self.similarity_features[candidate_indices]
        sims = np.dot(candidate_feats, query_feat)
        ranked_pos = np.argsort(-sims)
        ranked_names = [candidate_pairs[i][0] for i in ranked_pos]

        if len(ranked_names) >= total:
            return ranked_names[:total]
        if len(ranked_names) == 0:
            replace = len(candidates) < total
            return np.random.choice(candidates, total, replace=replace).tolist()
        if len(ranked_names) == 1:
            return ranked_names * total
        return ranked_names + np.random.choice(ranked_names, total - len(ranked_names), replace=True).tolist()

    def get_query_feature(self, query_name):
        if query_name not in self.query_feature_cache:
            query_img = self.read_img_array(query_name)
            self.query_feature_cache[query_name] = self.extract_similarity_feature(query_img)
        return self.query_feature_cache[query_name]

    def extract_similarity_feature(self, rgb):
        resized = np.array(Image.fromarray(rgb).resize((self.similarity_size, self.similarity_size), Image.BILINEAR))
        feat = resized.astype(np.float32).reshape(-1) / 255.0
        norm = np.linalg.norm(feat) + 1e-12
        return feat / norm

    def load_or_build_class_representativity(self, cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            class_repr = {int(k): {kk: float(vv) for kk, vv in v.items()} for k, v in cached.items()}
            if all(cls in class_repr for cls in self.class_ids):
                return class_repr

        entries = set()
        for cls in self.class_ids:
            entries.update(self.support_pool.get(cls, []))

        class_repr = {cls: {} for cls in self.class_ids}
        for entry in entries:
            mask = self.read_mask(entry).numpy()
            total = float(mask.size)
            if total == 0:
                continue
            for cls in self.class_ids:
                class_repr[cls][entry] = float((mask == cls).sum()) / total

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        serializable = {str(k): v for k, v in class_repr.items()}
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f)
        return class_repr

    def load_or_build_similarity_index(self, cache_path):
        if os.path.exists(cache_path):
            cache = np.load(cache_path)
            entries = cache['entries'].astype(str)
            features = cache['features'].astype(np.float32)
            index = {name: i for i, name in enumerate(entries.tolist())}
            return entries.tolist(), features, index

        entries = set(self.query_pool)
        for cls in self.class_ids:
            entries.update(self.support_pool.get(cls, []))
        entry_list = sorted(entries)

        features = []
        for entry in entry_list:
            rgb = self.read_img_array(entry)
            features.append(self.extract_similarity_feature(rgb))
        features = np.stack(features, axis=0).astype(np.float32)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, entries=np.array(entry_list), features=features)
        index = {name: i for i, name in enumerate(entry_list)}
        return entry_list, features, index
    
    
    # def build_img_metadata(self):

    #     def read_metadata(split, fold_id):  
    #         fold_n_metadata = os.path.join(f'data/splits/chesapeake/fold0.txt')
    #         with open(fold_n_metadata, 'r') as f:
    #             fold_n_metadata = f.read().split('\n')[:-1]
    #         fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1])] for data in fold_n_metadata]
    #         return fold_n_metadata

    #     img_metadata = []
    #     if self.split == 'trn':  # For training, read image-metadata of "the other" folds
    #         for fold_id in range(self.nfolds):
    #             if fold_id == self.fold:  # Skip validation fold
    #                 continue
    #             img_metadata += read_metadata(self.split, fold_id)
    #     elif self.split == 'val':  # For validation, read image-metadata of "current" fold
    #         img_metadata = read_metadata(self.split, self.fold)
    #     else:
    #         raise Exception('Undefined split %s: ' % self.split)

    #     print(f'Total {self.split} images are : {len(img_metadata):,}')

    #     return img_metadata

    # def build_img_metadata_classwise(self):
    #     img_metadata_classwise = {}
    #     for class_id in range(1, self.nclass + 1):
    #         img_metadata_classwise[class_id] = []

    #     for img_name, img_class in self.img_metadata:
    #         img_metadata_classwise[img_class] += [img_name]

    #     # img_metadata_classwise.keys(): [1, 2, ..., 20]
    #     assert 0 not in img_metadata_classwise.keys()
    #     assert self.nclass in img_metadata_classwise.keys()

    #     return img_metadata_classwise

