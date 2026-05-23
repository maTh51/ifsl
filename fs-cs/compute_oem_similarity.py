#!/usr/bin/env python3
"""Precompute the crop-aware OEM similarity cache.

This script reuses the OEM dataset implementation to build the similarity
index from the static dataset split definitions. The cache is built from the
cropped query/support tiles, not from the full 1024x1024 images.
"""

import argparse
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
	sys.path.insert(0, SCRIPT_DIR)


from data.oem import DatasetOEM


class SimilarityCacheOEM(DatasetOEM):
	def build_img_metadata_classwise(self):
		return {class_id: [] for class_id in self.class_ids}

	def build_img_metadata(self):
		return []


def parse_args():
	parser = argparse.ArgumentParser(description='Precompute OEM similarity features for crop-aware evaluation')
	parser.add_argument('--datapath', type=str, required=True,
					help='OEM dataset root containing train.txt, val.json, test.json, and split folders')
	parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
					help='Split to precompute the similarity cache for')
	parser.add_argument('--fold', type=int, default=0,
					help='OEM fold index used by the dataset loader')
	parser.add_argument('--way', type=int, default=4,
					help='Number of ways used to keep the same split prefix behavior as evaluation')
	parser.add_argument('--shot', type=int, default=1,
					help='Shot count passed to the dataset constructor')
	parser.add_argument('--bgd', action='store_true',
					help='Enable background class handling to match evaluation settings')
	parser.add_argument('--rdn_sup', action='store_true',
					help='Use random supports to match evaluation settings when desired')
	parser.add_argument('--oem_train_list', type=str, default='train.txt',
					help='OEM train list path, absolute or relative to --datapath')
	parser.add_argument('--oem_val_json', type=str, default='val.json',
					help='OEM validation JSON path, absolute or relative to --datapath')
	parser.add_argument('--oem_test_json', type=str, default='test.json',
					help='OEM test JSON path, absolute or relative to --datapath')
	parser.add_argument('--oem_val_pools', type=str, default='oem_val_pools.json',
					help='OEM query-tile pool JSON path')
	parser.add_argument('--oem_train_pools', type=str, default='oem_train_pools.json',
					help='OEM support-tile pool JSON path')
	parser.add_argument('--oem_sw_tile', type=int, default=400,
					help='Crop size used for sliding-window query tiles')
	parser.add_argument('--oem_sw_stride', type=int, default=312,
					help='Stride used for sliding-window query tiles')
	parser.add_argument('--oem_crop_size', type=int, default=400,
					help='Training crop size passed through to the dataset')
	parser.add_argument('--support_similarity_size', type=int, default=32,
					help='Feature resize used for similarity embeddings')
	parser.add_argument('--support_similarity_cache', type=str, default='auto',
					help='Cache path or auto for the similarity index file')
	parser.add_argument('--force', action='store_true',
					help='Force recompute even if the cache already exists')
	return parser.parse_args()


def main():
	args = parse_args()
	json_path = os.path.join(args.datapath, args.oem_val_json if args.split == 'val' else args.oem_test_json)
	with open(json_path, 'r', encoding='utf-8') as f:
		split_data = json.load(f)
	eval_classes = sorted({int(cls) for cls in split_data.get('support_set', {}).keys()})
	if not eval_classes:
		raise RuntimeError(f'No support_set classes found in {json_path}')
	dataset = SimilarityCacheOEM(
		datapath=args.datapath,
		fold=args.fold,
		transform=None,
		split=args.split,
		way=args.way,
		shot=args.shot,
		bgd=args.bgd,
		rdn_sup=args.rdn_sup,
		oem_train_list=args.oem_train_list,
		oem_val_json=args.oem_val_json,
		oem_test_json=args.oem_test_json,
		oem_crop_size=args.oem_crop_size,
		support_strategy='similarity',
		support_similarity_cache=args.support_similarity_cache,
		support_similarity_size=args.support_similarity_size,
		use_sliding_window=True,
		eval_classes=eval_classes,
		oem_val_pools=args.oem_val_pools,
		oem_train_pools=args.oem_train_pools,
		oem_sw_tile=args.oem_sw_tile,
		oem_sw_stride=args.oem_sw_stride,
	)

	cache_path = dataset.resolve_cache_path(
		args.support_similarity_cache,
		f'oem_similarity_index_tiles_v2_{args.split}.npz',
	)
	if args.force and os.path.exists(cache_path):
		os.remove(cache_path)

	entries, features, index = dataset.load_or_build_similarity_index(cache_path)
	print(f'[oem-similarity] split={args.split}')
	print(f'[oem-similarity] cache={cache_path}')
	print(f'[oem-similarity] entries={len(entries)} features={features.shape}')
	print(f'[oem-similarity] index_size={len(index)}')


if __name__ == '__main__':
	main()