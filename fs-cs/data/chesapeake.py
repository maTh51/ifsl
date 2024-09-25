r""" Chesapeake few-shot classification and segmentation dataset """
import os

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
    def __init__(self, datapath, fold, transform, split, way, shot, bgclass, bgd, rdn_sup):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.nfolds = 1
        self.nclass = 6 - (1 if bgclass != 0 else -0)
        self.benchmark = 'chesapeake'

        self.fold = fold
        self.way = self.nclass
        self.shot = shot
        self.bgclass = bgclass

        self.img_path = os.path.join(datapath)
        self.ann_path = os.path.join(datapath)
        # self.pool_path = os.path.join(datapath, 'chesapeake_400/pools/')
        self.pool_path = '/home/matheuspimenta/Jobs/SR/ifsl/utils'
        self.transform = transform

        self.bgd = bgd
        # if not self.bgd:
        #     self.mask_palette = {
        #         1: (255, 255, 255),  # Impervious surfaces (white)
        #         2: (0, 0, 255),  # Buildings (blue)
        #         3: (0, 255, 255),  # Low vegetation (cyan)
        #         4: (0, 255, 0),  # Trees (green)
        #         5: (255, 255, 0),  # Cars (yellow)
        #         6: (255, 0, 0),  # Clutter (red)
        #         0: (0, 0, 0)  # Undefined (black)
        #     }
        # else:
        #     self.mask_palette = {
        #         0:  0,    # não existe no dataset
        #         1:  1,    # agua
        #         2:  2,    # floresta
        #         3:  3,    # campo
        #         4:  4,    # terra estéril
        #         5:  5,    # impermeável (outro)
        #         6:  6,    # impermeável (estrada)
        #         15: 0,   # sem dados
        #     }


        # self.class_ids = self.build_class_ids()
        # self.img_metadata = self.build_img_metadata()
        # self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.class_ids = [1,2,3,4,5,6]
        if self.bgclass > 0:
            self.class_ids.remove(self.bgclass)
        self.query_pool, self.support_pool = self.build_pools() 
        self.rdn_sup = rdn_sup

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else len(self.query_pool)

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
            print("@@@@@@@@@")
            print(query_name)
            # print(torch.unique(query_mask))
            # print(query_class_presence.int().sum())
            # print((len(torch.unique(query_mask)) - 1))
            print(query_class_presence)
            print(torch.unique(query_mask))
            print("#######")
            print(query_class_presence.int().sum())
            print((len(torch.unique(query_mask))))
            print(1 if (self.bgd and 0 in torch.unique(query_mask)) else 0)
            print("**************************************************************")
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
        # return Image.open(os.path.join(self.img_path, img_name))
        path, dims = img_name.split(";")
        dims = dims.split(",")
        window = Window(int(dims[0]), int(dims[1]), 400, 400)
        with rio.open(os.path.join(self.img_path, path.replace( "lc.tif", "naip-new.tif"))) as src:
            red = src.read(1, window=window)
            green = src.read(2, window=window)
            blue = src.read(3, window=window)
            rgb = np.dstack((red, green, blue))
        return Image.fromarray(rgb, 'RGB')
    
    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        # Image.open(os.path.join(self.ann_path, img_name))
        path, dims = img_name.split(";")
        dims = dims.split(",")
        window = Window(int(dims[0]), int(dims[1]), 400, 400)
        rast_mask = rio.open(os.path.join(self.ann_path, path)).read(1, window=window)
        # mask = torch.tensor(self.convert_mask(np.moveaxis(rast_mask, 0, -1)))
        if self.bgclass != 0:
            cv_rast_mask = self.convert_mask(rast_mask)
        return torch.tensor(cv_rast_mask)

    def convert_mask(self, np_mask):
        # height, width = np_mask.shape
        
        # # Cria um novo array de zeros com a mesma altura e largura
        # new_mask = np.zeros((height, width), dtype=np.uint8)
        
        # # Mapeia os valores do primeiro canal para novas classes usando o dicionário mask_palette
        # unique_classes = np.unique(np_mask)
        # mask_palette_values = np.array([self.mask_palette.get(c, 0) for c in unique_classes], dtype=np.uint8)
        # class_to_color = dict(zip(unique_classes, mask_palette_values))
        
        # # Usa a função np.vectorize para mapear as classes para as novas cores
        # vectorized_mapping = np.vectorize(class_to_color.get)
        # new_mask = vectorized_mapping(np_mask)
        
        # return new_mask
        transformed_mask = np.copy(np_mask)
        if self.bgclass > 0:
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print(np.unique(transformed_mask))
            transformed_mask[transformed_mask == self.bgclass] = 0
            # print(np.unique(transformed_mask))
            transformed_mask[transformed_mask > self.bgclass] -= 1
            # print(np.unique(transformed_mask))
            # print("#############################################")
            # time.sleep(1)
        return transformed_mask

    def sample_episode(self, idx):
        # Fix (q, s) pair for all queries across different batch sizes for reproducibility
        if self.split == 'val':
            np.random.seed(idx)

        idx %= len(self.query_pool)  # for testing, as n_images < 1000
        query_name = self.query_pool[idx]
        # black_list = ['top_mosaic_09cm_area30/part9.png','top_mosaic_09cm_area15/part2.png','top_mosaic_09cm_area15/part13.png']
        # while query_name in black_list:
        #     idx = idx + 1
        #     idx %= len(self.query_pool)
        #     query_name = self.query_pool[idx]


        # 3-way 2-shot support_names: [[c1_1, c1_2], [c2_1, c2_2], [c3_1, c3_2]]
        support_names = []
        support_names_c = []

        # p encourage the support classes sampled as the query_class by the prob of 0.5
        # p = np.ones([len(self.class_ids)]) / 2. / float(len(self.class_ids) - 1)
        # p[self.class_ids.index(query_class)] = 1 / 2.
        # support_classes = np.random.choice(self.class_ids, self.way, p=p, replace=False).tolist()
        
        support_classes = self.class_ids # Todas as classes possíveis menos vazio
        if self.rdn_sup:
            for sc in range(len(self.class_ids) (+1 if self.bgclass != 0 else +0)):
                support_names_c = []
                while True:
                    support_name = np.random.choice(self.support_pool[21], 1, replace=False)[0]
                    if query_name != support_name:
                        support_names_c.append(support_name)
                    if len(support_names_c) == self.shot:
                        break
                support_names.append(support_names_c)
        else:
            for sc in support_classes:
                support_names_c = []
                while True:  # keep sampling support set if query == support
                    if len(self.support_pool[sc]) == 0:
                        print("ERROR!!")
                        print(sc)
                        break
                    support_name = np.random.choice(self.support_pool[sc], 1, replace=False)[0]
                    if query_name != support_name and support_name not in support_names_c:
                        support_names_c.append(support_name)
                    if len(support_names_c) == self.shot:
                        break
                support_names.append(support_names_c)

        return query_name, support_names, [i for i in range(1,self.way+1)]

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
        query_file = os.path.join(self.pool_path, f'4.txt')
        with open(query_file, 'r') as f:
            query_pool = f.read().split('\n')[:-1]

        support_pool = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            21: [], # todas
            30: [], # selecionadas
        }
        for i in self.class_ids:
            support_file = os.path.join(self.pool_path, f'{i}.txt')
            with open(support_file, 'r') as f:
                support_pool[i] = f.read().split('\n')[:-1]
                # print(i)
                # print(len(support_pool[i]))
                support_pool[21] += support_pool[i]
                selected_images = random.sample(support_pool[i], 5)
                support_pool[30] += selected_images

        for v in support_pool.values():
            print(len(v))
        # return query_pool, support_pool
        # return random.sample(support_pool[21], 100), support_pool
        return support_pool[30], support_pool
    
    
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

