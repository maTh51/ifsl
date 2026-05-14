r""" Dataloader builder for few-shot classification and segmentation task """
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.vaihingen import DatasetVAIHINGEN
from data.chesapeake import DatasetCHESAPEAKE
from data.oem import DatasetOEM



class FSCSDatasetModule(LightningDataModule):
    """
    A LightningDataModule for FS-CS benchmark
    """
    def __init__(self, args, img_size=400):
        super().__init__()
        self.args = args
        self.datapath = args.datapath

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        self.img_size = img_size
        self.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'vaihingen': DatasetVAIHINGEN,
            'chesapeake': DatasetCHESAPEAKE,
            'oem': DatasetOEM,
        }
        self.transform = transforms.Compose([transforms.Resize(size=(self.img_size, self.img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.img_mean, self.img_std)])

    def train_dataloader(self):
        kwargs = {'datapath': self.datapath,
                  'fold': self.args.fold,
                  'transform': self.transform,
                  'split': 'trn',
                  'way': self.args.way,
                  'shot': 1}  # shot=1 fixed for training
        # Add dataset-specific parameters for Chesapeake and Vaihingen
        if self.args.benchmark in ['chesapeake', 'vaihingen']:
            kwargs.update({'bgclass': self.args.bgclass,
                           'bgd': self.args.bgd,
                           'rdn_sup': self.args.rdn_sup})
            if self.args.benchmark == 'vaihingen':
                kwargs.update({'merge_class': self.args.merge_class})
            if self.args.benchmark == 'chesapeake':
                kwargs.update({'support_strategy': self.args.support_strategy,
                               'support_area_cache': self.args.support_area_cache,
                               'support_similarity_cache': self.args.support_similarity_cache,
                               'support_similarity_size': self.args.support_similarity_size})
        elif self.args.benchmark == 'oem':
            kwargs.update({'bgd': self.args.bgd,
                           'rdn_sup': self.args.rdn_sup,
                           'oem_train_list': self.args.oem_train_list,
                           'oem_val_json': self.args.oem_val_json,
                           'oem_test_json': self.args.oem_test_json,
                           'oem_crop_size': self.args.oem_crop_size,
                           'support_strategy': self.args.support_strategy,
                           'support_area_cache': self.args.support_area_cache,
                           'support_similarity_cache': self.args.support_similarity_cache,
                           'support_similarity_size': self.args.support_similarity_size})
        dataset = self.datasets[self.args.benchmark](**kwargs)
        dataloader = DataLoader(dataset, batch_size=self.args.bsz, shuffle=True, num_workers=8)
        return dataloader

    def val_dataloader(self):
        kwargs = {'datapath': self.datapath,
                  'fold': self.args.fold,
                  'transform': self.transform,
                  'split': 'val',
                  'way': self.args.way,
                  'shot': self.args.shot}
        if self.args.benchmark in ['chesapeake', 'vaihingen']:
            kwargs.update({'bgclass': self.args.bgclass,
                           'bgd': self.args.bgd,
                           'rdn_sup': self.args.rdn_sup})
            if self.args.benchmark == 'vaihingen':
                kwargs.update({'merge_class': self.args.merge_class})
            if self.args.benchmark == 'chesapeake':
                kwargs.update({'support_strategy': self.args.support_strategy,
                               'support_area_cache': self.args.support_area_cache,
                               'support_similarity_cache': self.args.support_similarity_cache,
                               'support_similarity_size': self.args.support_similarity_size})
        elif self.args.benchmark == 'oem':
            kwargs.update({'bgd': self.args.bgd,
                           'rdn_sup': self.args.rdn_sup,
                           'oem_train_list': self.args.oem_train_list,
                           'oem_val_json': self.args.oem_val_json,
                           'oem_test_json': self.args.oem_test_json,
                           'oem_crop_size': self.args.oem_crop_size,
                           'support_strategy': self.args.support_strategy,
                           'support_area_cache': self.args.support_area_cache,
                           'support_similarity_cache': self.args.support_similarity_cache,
                           'support_similarity_size': self.args.support_similarity_size})

        dataset = self.datasets[self.args.benchmark](**kwargs)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)
        return dataloader

    def test_dataloader(self):
        kwargs = {'datapath': self.datapath,
                  'fold': self.args.fold,
                  'transform': self.transform,
                  'split': 'test' if self.args.benchmark == 'oem' else 'val',
                  'way': self.args.way,
                  'shot': self.args.shot}

        if self.args.benchmark in ['chesapeake', 'vaihingen']:
            kwargs.update({'bgclass': self.args.bgclass,
                           'bgd': self.args.bgd,
                           'rdn_sup': self.args.rdn_sup})
            if self.args.benchmark == 'vaihingen':
                kwargs.update({'merge_class': self.args.merge_class})
            if self.args.benchmark == 'chesapeake':
                kwargs.update({'support_strategy': self.args.support_strategy,
                               'support_area_cache': self.args.support_area_cache,
                               'support_similarity_cache': self.args.support_similarity_cache,
                               'support_similarity_size': self.args.support_similarity_size})
        elif self.args.benchmark == 'oem':
            kwargs.update({'bgd': self.args.bgd,
                           'rdn_sup': self.args.rdn_sup,
                           'oem_train_list': self.args.oem_train_list,
                           'oem_val_json': self.args.oem_val_json,
                           'oem_test_json': self.args.oem_test_json,
                           'oem_crop_size': self.args.oem_crop_size,
                           'support_strategy': self.args.support_strategy,
                           'support_area_cache': self.args.support_area_cache,
                           'support_similarity_cache': self.args.support_similarity_cache,
                           'support_similarity_size': self.args.support_similarity_size})

        dataset = self.datasets[self.args.benchmark](**kwargs)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)
        return dataloader
