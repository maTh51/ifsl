import torch
import argparse

from pytorch_lightning import Trainer

from model.asnet import AttentiveSqueezeNetwork
from model.panet import PrototypeAlignmentNetwork
from model.pfenet import PriorGuidedFeatureEnrichmentNetwork
from model.hsnet import HypercorrSqueezeNetwork
from model.asnethm import AttentiveSqueezeNetworkHM
from data.dataset import FSCSDatasetModule
from common.callbacks import MeterCallback, CustomProgressBar, CustomCheckpoint, OnlineLogger

import os
import GPUtil


def main(args):

    # Method
    modeldict = dict(panet=PrototypeAlignmentNetwork,
                     pfenet=PriorGuidedFeatureEnrichmentNetwork,
                     hsnet=HypercorrSqueezeNetwork,
                     asnethm=AttentiveSqueezeNetworkHM,
                     asnet=AttentiveSqueezeNetwork)
    modelclass = modeldict[args.method]

    # Dataset initialization
    dm = FSCSDatasetModule(args)

    # Pytorch-lightning main trainer
    checkpoint_callback = CustomCheckpoint(args)
    trainer = Trainer(accelerator='dp',  # DataParallel
                      callbacks=[MeterCallback(args), CustomCheckpoint(args), CustomProgressBar()],
                      gpus=torch.cuda.device_count(),
                      logger=False if args.nowandb or args.eval else OnlineLogger(args),
                      progress_bar_refresh_rate=1,
                      max_epochs=args.niter,
                      num_sanity_val_steps=0,
                      weights_summary=None,
                      resume_from_checkpoint=checkpoint_callback.lastmodelpath,
                      # profiler='advanced',  # this awesome profiler is easy to use
                      )

    if args.eval:
        # Loading the best model checkpoint from args.logpath
        modelpath = checkpoint_callback.modelpath
        model = modelclass.load_from_checkpoint(modelpath, args=args)
        trainer.test(model, test_dataloaders=dm.test_dataloader())
    else:
        # Train
        model = modelclass(args)
        trainer.fit(model, dm)


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Methods for Integrative Few-Shot Classification and Segmentation')
    parser.add_argument('--datapath', type=str, default='~/datasets', help='Dataset path containing the root dir of pascal & coco')
    parser.add_argument('--method', type=str, default='asnet', choices=['panet', 'pfenet', 'hsnet', 'asnet', 'asnethm'], help='FS-CS methods')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'vaihingen', 'chesapeake', 'oem'], help='Experiment benchmark')
    parser.add_argument('--logpath', type=str, default='', help='Checkpoint saving dir identifier')
    parser.add_argument('--ckptpath', type=str, default='', help='Checkpoint file or directory to load when using --eval')
    parser.add_argument('--way', type=int, default=1, help='N-way for K-shot evaluation episode')
    parser.add_argument('--shot', type=int, default=1, help='K-shot for N-way K-shot evaluation episode: fixed to 1 for training')
    parser.add_argument('--bsz', type=int, default=12, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--niter', type=int, default=2000, help='Max iterations')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3], help='4-fold validation fold')
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet50', 'resnet101'], help='Backbone CNN network')
    parser.add_argument('--nowandb', action='store_true', help='Flag not to log at wandb')
    parser.add_argument('--eval', action='store_true', help='Flag to evaluate a model checkpoint')
    parser.add_argument('--weak', action='store_true', help='Flag to train with cls (weak) labels -- reduce learning rate by 10 times')
    parser.add_argument('--resume', action='store_true', help='Flag to resume a finished run')
    parser.add_argument('--vis', action='store_true', help='Flag to visualize. Use with --eval')
    
    parser.add_argument('--bgd', action='store_true', help='With background?')
    parser.add_argument('--rdn_sup', action='store_true', help='Random support images?')
    parser.add_argument('--bgclass', type=int, default=0, help='Background Class')
    parser.add_argument('--merge_class', type=int, default=None, choices=[1, 2, 3, 4, 5, 6],
                        help='Vaihingen class ID to merge into class 1 (Imp. Surfaces)')
    parser.add_argument('--support_strategy', '--oem_support_strategy', dest='support_strategy', type=str,
                        default='random', choices=['random', 'max_area', 'similarity'],
                        help='Support sampling strategy at evaluation time')
    parser.add_argument('--support_area_cache', '--oem_area_cache', dest='support_area_cache', type=str,
                        default='auto', help='Path to class representativity cache JSON (or auto)')
    parser.add_argument('--support_similarity_cache', '--oem_similarity_cache', dest='support_similarity_cache', type=str,
                        default='auto', help='Path to similarity feature index NPZ (or auto)')
    parser.add_argument('--support_similarity_size', '--oem_similarity_size', dest='support_similarity_size', type=int,
                        default=32, help='Image resize used to build support similarity feature vectors')
    parser.add_argument('--oem_train_list', type=str, default='train.txt',
                        help='OEM train list path (absolute or relative to --datapath)')
    parser.add_argument('--oem_val_json', type=str, default='val.json',
                        help='OEM validation JSON path (absolute or relative to --datapath)')
    parser.add_argument('--oem_test_json', type=str, default='test.json',
                        help='OEM test JSON path (absolute or relative to --datapath)')
    parser.add_argument('--oem_crop_size', type=int, default=400,
                        help='OEM random crop size used in training episodes')
    parser.add_argument('--oem_sw_enable', action='store_true',
                        help='Enable OEM sliding-window query inference at test time')
    parser.add_argument('--oem_sw_tile', type=int, default=400,
                        help='OEM sliding-window tile size')
    parser.add_argument('--oem_sw_stride', type=int, default=312,
                        help='OEM sliding-window stride')
    
    args = parser.parse_args()

    # Encontra a GPU com mais memória livre
    gpus = GPUtil.getGPUs()
    least_used_gpu = min(gpus, key=lambda gpu: gpu.memoryUsed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(least_used_gpu.id)

    main(args)
