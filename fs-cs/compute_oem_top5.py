#!/usr/bin/env python3
"""Precompute per-query top-K similar support tiles (class-agnostic).

Produces a JSON mapping from query tile signature -> list of top-K support
tile signatures (and similarity scores). Reuses the tile-aware similarity
NPZ produced by compute_oem_similarity.py when available.
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np

from data.oem import DatasetOEM


class SimilarityCacheOEM(DatasetOEM):
    def build_img_metadata_classwise(self):
        return {class_id: [] for class_id in self.class_ids}

    def build_img_metadata(self):
        return []


def build_sliding_tiles(dataset, image_entry, split_name):
    """Build 400x400 tiles for a single image entry."""
    image_name = dataset.entry_image_name(image_entry)
    image = dataset.read_img(image_name)
    width, height = image.size
    if height < dataset.sliding_tile_size or width < dataset.sliding_tile_size:
        return []

    tiles = []
    for y in dataset._sliding_positions(height, dataset.sliding_tile_size, dataset.sliding_stride):
        for x in dataset._sliding_positions(width, dataset.sliding_tile_size, dataset.sliding_stride):
            tiles.append({
                'split': split_name,
                'image_name': os.path.basename(image_name),
                'x': int(x),
                'y': int(y),
                'size': int(dataset.sliding_tile_size),
            })
    return tiles


def parse_args():
    p = argparse.ArgumentParser(description='Build top-K similar support list per query tile')
    p.add_argument('--datapath', required=True, help='OEM dataset root (contains pools/ and val.json)')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--oem_val_json', default='val.json')
    p.add_argument('--oem_val_pools', default='oem_val_pools.json')
    p.add_argument('--oem_train_pools', default='oem_train_pools.json')
    p.add_argument('--support_similarity_cache', default='auto', help='Path or "auto" for NPZ cache')
    p.add_argument('--support_similarity_size', type=int, default=32)
    p.add_argument('--oem_sw_tile', type=int, default=400)
    p.add_argument('--oem_sw_stride', type=int, default=312)
    p.add_argument('--way', type=int, default=4)
    p.add_argument('--shot', type=int, default=1)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--out', default=None, help='Output JSON path (default: pools/oem_similarity_topK_{split}.json)')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    json_path = os.path.join(args.datapath, args.oem_val_json if args.split == 'val' else args.oem_val_json)
    with open(json_path, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    eval_classes = sorted({int(cls) for cls in split_data.get('support_set', {}).keys()})
    if not eval_classes:
        raise RuntimeError(f'No support_set classes found in {json_path}')

    dataset = SimilarityCacheOEM(
        datapath=args.datapath,
        fold=0,
        transform=None,
        split=args.split,
        way=args.way,
        shot=args.shot,
        bgd=False,
        rdn_sup=False,
        oem_val_pools=args.oem_val_pools,
        oem_train_pools=args.oem_train_pools,
        oem_sw_tile=args.oem_sw_tile,
        oem_sw_stride=args.oem_sw_stride,
        support_similarity_cache=args.support_similarity_cache,
        support_similarity_size=args.support_similarity_size,
        use_sliding_window=True,
        eval_classes=eval_classes,
    )

    cache_path = dataset.resolve_cache_path(
        args.support_similarity_cache, f'oem_similarity_index_tiles_v2_{args.split}.npz')
    if args.force and os.path.exists(cache_path):
        os.remove(cache_path)

    entries, features, index = dataset.load_or_build_similarity_index(cache_path)

    # Normalize entries -> signatures
    if len(entries) and isinstance(entries[0], dict):
        entry_sigs = [dataset.entry_signature(e) for e in entries]
    else:
        entry_sigs = [str(e) for e in entries]

    # Build support list (class-agnostic). Prefer combined pool if available,
    # then add tiled val/test support images so they participate too.
    if isinstance(dataset.support_pool, dict) and 21 in dataset.support_pool and dataset.support_pool[21]:
        support_entries = list(dataset.support_pool[21])
    else:
        support_entries = []
        for k, v in dataset.support_pool.items():
            if isinstance(v, list):
                support_entries.extend(v)

    support_json_entries = []
    support_set = split_data.get('support_set', {})
    for class_key, names in support_set.items():
        if not isinstance(names, list):
            continue
        for name in names:
            image_entry = os.path.join('valset' if args.split == 'val' else 'testset', 'labels', name).replace('\\', '/')
            if os.path.exists(os.path.join(args.datapath, image_entry)):
                support_json_entries.extend(build_sliding_tiles(dataset, image_entry, 'valset' if args.split == 'val' else 'testset'))

    support_entries.extend(support_json_entries)

    # Also try to load explicit train pools file and add those tiles as trainset entries
    train_pools_path = dataset.resolve_split_path(args.oem_train_pools, args.oem_train_pools)
    if os.path.exists(train_pools_path):
        try:
            with open(train_pools_path, 'r', encoding='utf-8') as f:
                train_pools = json.load(f)
            for cls, tiles in train_pools.items():
                if not isinstance(tiles, list):
                    continue
                for t in tiles:
                    if isinstance(t, dict):
                        t = dict(t)
                        t.setdefault('split', 'trainset')
                        support_entries.append(t)
                    else:
                        # string entry; convert to trainset labels path
                        entry = os.path.join('trainset', 'labels', os.path.basename(t)).replace('\\', '/')
                        support_entries.append(entry)
        except Exception:
            pass
    else:
        # If explicit train pools file is missing, try tiling images listed in train.txt
        train_list_path = dataset.train_list_path if hasattr(dataset, 'train_list_path') else None
        if train_list_path and os.path.exists(train_list_path):
            try:
                with open(train_list_path, 'r', encoding='utf-8') as f:
                    train_names = [line.strip() for line in f if line.strip()]
                for name in train_names:
                    image_entry = os.path.join('trainset', 'labels', name).replace('\\', '/')
                    full_path = os.path.join(args.datapath, image_entry)
                    if os.path.exists(full_path):
                        support_entries.extend(build_sliding_tiles(dataset, image_entry, 'trainset'))
            except Exception:
                pass

    # Dedupe supports by signature while preserving an example entry object
    supports_by_sig = {}
    for e in support_entries:
        sig = dataset.entry_signature(e)
        if sig not in supports_by_sig:
            supports_by_sig[sig] = e

    support_sigs = list(supports_by_sig.keys())

    # Assemble support features (use cached features when available)
    support_feats = []
    for sig in support_sigs:
        if sig in index:
            support_feats.append(features[index[sig]])
        else:
            entry_obj = supports_by_sig[sig]
            rgb = dataset.entry_similarity_rgb(entry_obj)
            support_feats.append(dataset.extract_similarity_feature(rgb))
    support_feats = np.stack(support_feats, axis=0).astype(np.float32)

    # Iterate over query tiles and compute top-K supports
    results = {}
    query_tiles = dataset.query_tiles if getattr(dataset, 'query_tiles', None) else dataset.query_pool
    for tile in query_tiles:
        qsig = dataset.entry_signature(tile)
        # get query feature (prefer cache/index)
        if qsig in index:
            qfeat = features[index[qsig]]
        else:
            # compute via get_query_feature using image name + coords
            qname = dataset.entry_image_name(tile)
            qcoords = dataset.entry_coords(tile)
            qfeat = dataset.get_query_feature(qname, tile_coords=qcoords)

        sims = np.dot(support_feats, qfeat)
        # sort desc
        order = np.argsort(-sims)[: args.top_k]
        top_list = []
        for idx in order:
            top_list.append({'support': support_sigs[int(idx)], 'score': float(sims[int(idx)])})
        results[qsig] = top_list

    out_path = args.out or os.path.join(args.datapath, 'pools', f'oem_similarity_top{args.top_k}_{args.split}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f)

    print(f'Wrote top-{args.top_k} similarity mapping for {len(results)} queries to {out_path}')


if __name__ == '__main__':
    main()
