r""" OEM few-shot classification and segmentation dataset """
import os
import json

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import rasterio as rio
import random


class DatasetOEM(Dataset):
	"""FS-CS OEM dataset using official train/val/test split definitions."""

	def __init__(
		self,
		datapath,
		fold,
		transform,
		split,
		way,
		shot,
		bgd,
		rdn_sup,
		oem_train_list='train.txt',
		oem_val_json='val.json',
		oem_test_json='test.json',
		oem_crop_size=400,
		support_strategy='random',
		support_area_cache='auto',
		support_similarity_cache='auto',
		support_similarity_topk_cache='auto',
		support_similarity_size=32,
		use_sliding_window=False,
		eval_classes=None,
		oem_val_pools='oem_val_pools.json',
		oem_train_pools='oem_train_pools.json',
		oem_sw_tile=400,
		oem_sw_stride=312,
	):
		if split not in ['trn', 'val', 'test']:
			raise ValueError(f'Unsupported split={split}')
		self.split = split
		self.nfolds = 1
		self.nclass = 11
		self.benchmark = 'oem'

		self.fold = fold
		self.way = way
		self.shot = shot

		self.img_path = os.path.join(datapath)
		self.ann_path = os.path.join(datapath)
		self.pool_path = os.path.join(datapath, 'pools')
		self.transform = transform
		self.crop_size = int(oem_crop_size)
		self.train_list_path = self.resolve_split_path(oem_train_list, 'train.txt')
		self.val_json_path = self.resolve_split_path(oem_val_json, 'val.json')
		self.test_json_path = self.resolve_split_path(oem_test_json, 'test.json')
		self.query_has_mask = {}

		self.bgd = bgd
		self.use_sliding_window = use_sliding_window
		self.eval_classes = eval_classes
		self.support_sliding_window_pools = None
		self.sliding_tile_size = int(oem_sw_tile)
		self.sliding_stride = int(oem_sw_stride)
		self.query_tiles = []

		self.class_ids = self.build_class_ids(eval_classes)
		self.active_class_ids = self.class_ids
		if self.split in ['val', 'test'] and self.eval_classes is not None:
			self.active_class_ids = self.class_ids[:self.way]
		if self.way > len(self.class_ids):
			raise ValueError(f'way={self.way} is larger than available classes={len(self.class_ids)} for split={self.split}')

		if self.use_sliding_window and self.split in ['val', 'test']:
			self.support_sliding_window_pools = self.load_sliding_window_pools(oem_train_pools, default_split='trainset')

		self.query_pool, self.support_pool = self.build_pools()
		if self.use_sliding_window and self.split in ['val', 'test']:
			self.query_tiles = self.build_query_tiles(self.query_pool)
		self.img_metadata_classwise = self.build_img_metadata_classwise()
		self.img_metadata = self.build_img_metadata()
		self.rdn_sup = rdn_sup

		self.support_strategy = support_strategy
		valid_strategies = {'random', 'max_area', 'min_area', 'similarity', 'similarity_global'}
		if self.support_strategy not in valid_strategies:
			raise ValueError(f'Invalid support_strategy={self.support_strategy}. Choices: {sorted(valid_strategies)}')

		self.class_representativity = None
		self.similarity_entries = None
		self.similarity_features = None
		self.similarity_index = None
		self.similarity_topk = None
		self.query_feature_cache = {}
		self.similarity_size = int(support_similarity_size)
		self.support_similarity_topk_cache = support_similarity_topk_cache

		if self.split in ['val', 'test'] and self.support_strategy == 'max_area':
			area_cache_path = self.resolve_cache_path(support_area_cache, f'oem_class_representativity_{self.split}.json')
			self.class_representativity = self.load_or_build_class_representativity(area_cache_path)

		if self.split in ['val', 'test'] and self.support_strategy == 'min_area' and self.class_representativity is None:
			area_cache_path = self.resolve_cache_path(support_area_cache, f'oem_class_representativity_{self.split}.json')
			self.class_representativity = self.load_or_build_class_representativity(area_cache_path)

		if self.split in ['val', 'test'] and self.support_strategy in ('similarity', 'similarity_global'):
			topk_cache_path = self.resolve_cache_path(support_similarity_topk_cache, f'oem_similarity_top5_{self.split}.json')
			self.similarity_topk = self.load_similarity_topk(topk_cache_path)
			if self.similarity_topk is None:
				sim_cache_path = self.resolve_cache_path(support_similarity_cache, f'oem_similarity_index_tiles_v2_{self.split}.npz')
				(self.similarity_entries,
				 self.similarity_features,
				 self.similarity_index) = self.load_or_build_similarity_index(sim_cache_path)

	def __len__(self):
		if self.use_sliding_window and self.split in ['val', 'test']:
			return len(self.query_tiles)
		if self.split == 'trn':
			return len(self.img_metadata)
		if self.split == 'val':
			return max(1000, len(self.img_metadata))
		return len(self.img_metadata)

	def __getitem__(self, idx):
		# tile_coords is None for full-image episodes.
		# In OEM sliding-window eval mode, both query and supports are loaded as 400x400 tiles
		# using the coordinates from pool entries.
		tile_coords = None
		if self.use_sliding_window and self.split in ['val', 'test']:
			query_name, support_names, _support_classes, query_has_mask, tile_coords = self.sample_sliding_window_episode(idx)
			query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(
				query_name, support_names, query_has_mask, tile_coords=tile_coords
			)
		else:
			query_name, support_names, _support_classes, query_has_mask = self.sample_episode(idx)
			query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names, query_has_mask)

		if self.split == 'trn':
			query_img, query_cmask, support_imgs, support_cmasks = self.apply_random_crop_episode(query_img, query_cmask, support_imgs, support_cmasks)

		if query_has_mask:
			query_class_presence = [s_c in torch.unique(query_cmask) for s_c in _support_classes]
		else:
			query_class_presence = [False for _ in _support_classes]
		rename_class = lambda x: _support_classes.index(x) + 1 if x in _support_classes else 0

		query_img_raw = None
		# Only expose the full raw query image for sliding evaluation when we have the full image.
		if self.split in ['val', 'test'] and not (self.use_sliding_window and tile_coords is not None):
			query_img_raw = torch.from_numpy(np.array(query_img, dtype=np.uint8, copy=True)).permute(2, 0, 1)

		query_img = self.transform(query_img)
		query_mask, query_ignore_idx = self.get_query_mask(query_img, query_cmask, rename_class, query_has_mask)

		support_imgs = torch.stack([
			torch.stack([self.transform(support_img) for support_img in support_imgs_c])
			for support_imgs_c in support_imgs
		])
		support_masks, support_ignore_idxs = self.get_support_masks(support_imgs, _support_classes, support_cmasks, rename_class)

		_support_classes = torch.tensor(_support_classes)
		query_class_presence = torch.tensor(query_class_presence)

		if query_has_mask:
			non_zero_classes = [c for c in torch.unique(query_mask).tolist() if c != 0]
			assert query_class_presence.int().sum() == len(non_zero_classes)

		batch = {
			'query_img': query_img,
			'query_mask': query_mask,
			'query_name': query_name,
			'query_ignore_idx': query_ignore_idx,
			'has_query_mask': torch.tensor(query_has_mask, dtype=torch.bool),
			'org_query_imsize': org_qry_imsize,
			'support_imgs': support_imgs,
			'support_masks': support_masks,
			'support_names': support_names,
			'support_ignore_idxs': support_ignore_idxs,
			'support_classes': _support_classes,
			'query_class_presence': query_class_presence,
		}
		if query_img_raw is not None:
			batch['query_img_raw'] = query_img_raw

		return batch

	def get_query_mask(self, query_img, query_cmask, rename_class, query_has_mask):
		if not query_has_mask:
			spatial_size = query_img.size()[-2:]
			query_mask = torch.zeros(spatial_size, dtype=torch.long)
			query_ignore_idx = torch.ones(spatial_size, dtype=torch.bool)
			return query_mask, query_ignore_idx

		if self.split == 'trn':
			query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
		query_mask, query_ignore_idx = self.generate_query_episodic_mask(query_cmask.float(), rename_class)
		return query_mask, query_ignore_idx

	def get_support_masks(self, support_imgs, _support_classes, support_cmasks, rename_class):
		support_masks = []
		support_ignore_idxs = []
		for class_id, scmask_c in zip(_support_classes, support_cmasks):
			support_masks_c = []
			support_ignore_idxs_c = []
			for scmask in scmask_c:
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

	def load_frame(self, query_name, support_names, query_has_mask, tile_coords=None):
		query_img = self.read_img(query_name)
		query_mask = self.read_mask(query_name, allow_missing=not query_has_mask)
		if query_mask is None:
			query_mask = torch.zeros((query_img.size[1], query_img.size[0]), dtype=torch.long)
		# Keep size aligned with the returned query tensor/mask used by evaluation.
		org_qry_imsize = query_img.size
		if tile_coords is not None:
			x, y, tile_size = tile_coords['x'], tile_coords['y'], tile_coords['size']
			query_img, query_mask = self.crop_pair_at_coords(query_img, query_mask, x, y, tile_size)
			org_qry_imsize = query_img.size
		support_imgs = []
		support_masks = []
		for support_names_c in support_names:
			support_imgs_c = []
			support_masks_c = []
			for support_entry in support_names_c:
				support_name = self.entry_image_name(support_entry)
				support_img = self.read_img(support_name)
				support_mask = self.read_mask(support_name)
				if isinstance(support_entry, dict):
					support_img, support_mask = self.crop_pair_at_coords(
						support_img,
						support_mask,
						support_entry['x'],
						support_entry['y'],
						support_entry.get('size', self.crop_size),
					)
				support_imgs_c.append(support_img)
				support_masks_c.append(support_mask)
			support_imgs.append(support_imgs_c)
			support_masks.append(support_masks_c)

		return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

	def build_query_tiles(self, query_pool):
		query_tiles = []
		for entry in query_pool:
			query_name = self.entry_image_name(entry)
			query_img = self.read_img(query_name)
			width, height = query_img.size
			if height < self.sliding_tile_size or width < self.sliding_tile_size:
				continue

			ys = self._sliding_positions(height, self.sliding_tile_size, self.sliding_stride)
			xs = self._sliding_positions(width, self.sliding_tile_size, self.sliding_stride)
			for y in ys:
				for x in xs:
					query_tiles.append({
						'split': 'valset' if self.split == 'val' else 'testset',
						'image_name': os.path.basename(query_name),
						'x': int(x),
						'y': int(y),
						'size': int(self.sliding_tile_size),
					})
		return query_tiles

	def _sliding_positions(self, length, tile, stride):
		if length <= tile:
			return [0]
		positions = list(range(0, length - tile + 1, stride))
		if positions[-1] != length - tile:
			positions.append(length - tile)
		return positions

	def _entry_to_image_path(self, mask_entry):
		return mask_entry.replace('/labels/', '/images/')

	def read_img(self, mask_entry):
		r"""Return RGB image in PIL Image."""
		rgb = self.read_img_array(mask_entry)
		return Image.fromarray(rgb, 'RGB')

	def read_img_array(self, mask_entry):
		r"""Return RGB image as a numpy uint8 array."""
		img_rel_path = self._entry_to_image_path(mask_entry)
		with rio.open(os.path.join(self.img_path, img_rel_path)) as src:
			red = src.read(1)
			green = src.read(2)
			blue = src.read(3)
			rgb = np.dstack((red, green, blue))

		if rgb.dtype != np.uint8:
			rgb = np.clip(rgb, 0, 255).astype(np.uint8)
		return rgb

	def read_mask(self, mask_entry, allow_missing=False):
		r"""Return segmentation mask as tensor."""
		mask_path = os.path.join(self.ann_path, mask_entry)
		if allow_missing and not os.path.exists(mask_path):
			return None
		with rio.open(mask_path) as src:
			rast_mask = src.read(1)
		return torch.tensor(rast_mask)

	def apply_random_crop_episode(self, query_img, query_mask, support_imgs, support_masks):
		query_img, query_mask = self.random_crop_pair(query_img, query_mask, self.crop_size)
		support_imgs_cropped = []
		support_masks_cropped = []
		for imgs_c, masks_c in zip(support_imgs, support_masks):
			imgs_c_crop = []
			masks_c_crop = []
			for img, mask in zip(imgs_c, masks_c):
				img_crop, mask_crop = self.random_crop_pair(img, mask, self.crop_size)
				imgs_c_crop.append(img_crop)
				masks_c_crop.append(mask_crop)
			support_imgs_cropped.append(imgs_c_crop)
			support_masks_cropped.append(masks_c_crop)
		return query_img, query_mask, support_imgs_cropped, support_masks_cropped

	def random_crop_pair(self, img, mask, crop_size):
		w, h = img.size
		crop_h = min(int(crop_size), h)
		crop_w = min(int(crop_size), w)
		top = 0 if h == crop_h else random.randint(0, h - crop_h)
		left = 0 if w == crop_w else random.randint(0, w - crop_w)
		img_crop = img.crop((left, top, left + crop_w, top + crop_h))
		mask_crop = mask[top:top + crop_h, left:left + crop_w]
		return img_crop, mask_crop

	def crop_pair_at_coords(self, img, mask, x, y, tile_size):
		r"""Crop image and mask to specified coordinates and tile size."""
		w, h = img.size
		# Ensure we don't go out of bounds
		x1 = min(x, w - 1)
		y1 = min(y, h - 1)
		x2 = min(x + tile_size, w)
		y2 = min(y + tile_size, h)
		
		img_crop = img.crop((x1, y1, x2, y2))
		mask_crop = mask[y1:y2, x1:x2]
		return img_crop, mask_crop

	def entry_image_name(self, entry):
		if isinstance(entry, dict):
			split_dir = entry.get('split', 'valset' if self.split == 'val' else 'testset')
			return os.path.join(split_dir, 'labels', entry['image_name']).replace('\\', '/')
		return entry

	def entry_signature(self, entry):
		if isinstance(entry, dict):
			split_dir = entry.get('split', 'valset' if self.split == 'val' else 'testset')
			return f"{split_dir}::{entry['image_name']}::x={entry['x']}::y={entry['y']}::s={entry.get('size', self.crop_size)}"
		return entry

	def dedupe_entries(self, entries):
		seen = set()
		unique_entries = []
		for entry in entries:
			signature = self.entry_signature(entry)
			if signature in seen:
				continue
			seen.add(signature)
			unique_entries.append(entry)
		return unique_entries

	def entry_coords(self, entry):
		if isinstance(entry, dict):
			return {'x': int(entry['x']), 'y': int(entry['y']), 'size': int(entry.get('size', self.crop_size))}
		return None

	def entry_mask_path(self, entry):
		return self.entry_image_name(entry).replace('/images/', '/labels/')

	def entry_similarity_rgb(self, entry):
		rgb = self.read_img_array(self.entry_image_name(entry))
		coords = self.entry_coords(entry)
		if coords is None:
			return rgb
		crop_img = Image.fromarray(rgb, 'RGB')
		crop = self.crop_image_at_coords(crop_img, coords['x'], coords['y'], coords['size'])
		return np.array(crop, dtype=np.uint8, copy=True)

	def entry_class_fraction(self, entry, class_id):
		mask = self.read_mask(self.entry_mask_path(entry))
		coords = self.entry_coords(entry)
		if coords is not None:
			w = int(coords['size'])
			h = int(coords['size'])
			mask = mask[coords['y']:coords['y'] + h, coords['x']:coords['x'] + w]
		total = float(mask.numel())
		if total == 0:
			return 0.0
		return float((mask == class_id).sum().item()) / total

	def crop_image_at_coords(self, img, x, y, tile_size):
		w, h = img.size
		x1 = min(max(0, x), w - 1)
		y1 = min(max(0, y), h - 1)
		x2 = min(x1 + tile_size, w)
		y2 = min(y1 + tile_size, h)
		return img.crop((x1, y1, x2, y2))

	def sample_episode(self, idx):
		if self.split in ['val', 'test']:
			np.random.seed(idx)

		idx %= len(self.img_metadata)
		query_name, query_class, query_has_mask = self.img_metadata[idx]

		support_names = []
		# If using a global similarity strategy, precompute a global ranked list
		# of supports and split it into `way` chunks of `shot` each.
		global_similarity_chunks = None
		if self.support_strategy == 'similarity_global':
			total_needed = int(self.way) * int(self.shot)
			global_list = self.sample_similarity_global(None, query_name, total_needed)
			# split into equal chunks for each support class
			global_similarity_chunks = [global_list[i * int(self.shot):(i + 1) * int(self.shot)] for i in range(int(self.way))]
		if self.split in ['val', 'test'] and self.eval_classes is not None:
			support_classes = list(self.active_class_ids)
		elif self.way == 1:
			support_classes = [query_class]
		else:
			p = np.ones([len(self.class_ids)]) / 2. / float(len(self.class_ids) - 1)
			p[self.class_ids.index(query_class)] = 1 / 2.
			support_classes = np.random.choice(self.class_ids, self.way, p=p, replace=False).tolist()

		def sample_from_pool(pool, query_name, shot):
			# Prefer supports different from query. If the class pool is tiny,
			# fall back to replacement sampling to avoid dataloader deadlocks.
			candidates = [name for name in pool if name != query_name]
			if not candidates:
				candidates = list(pool)
			if not candidates:
				raise RuntimeError('Empty support candidates while sampling episode')
			replace = len(candidates) < shot
			return np.random.choice(candidates, shot, replace=replace).tolist()

		def select_support(pool, query_name, shot, class_id=None):
			if self.split == 'trn' or self.support_strategy == 'random':
				return sample_from_pool(pool, query_name, shot)
			if self.support_strategy == 'max_area' and class_id is not None:
				return self.sample_max_area_support(pool, query_name, shot, class_id)
			if self.support_strategy == 'similarity':
				return self.sample_similarity_support(pool, query_name, shot)
			if self.support_strategy == 'similarity_global':
				# This path should not be called per-class when global chunks are used.
				return sample_from_pool(pool, query_name, shot)
			return sample_from_pool(pool, query_name, shot)

		if self.rdn_sup:
			for _ in support_classes:
				support_names.append(select_support(self.support_pool[21], query_name, self.shot, class_id=None))
		else:
			if global_similarity_chunks is not None:
				# Assign precomputed global chunks to each support class in order.
				for chunk in global_similarity_chunks:
					support_names.append(chunk)
			else:
				for sc in support_classes:
					if len(self.support_pool[sc]) == 0:
						raise RuntimeError(f'Empty support pool for class {sc} in {self.pool_path}')
					support_names.append(select_support(self.support_pool[sc], query_name, self.shot, class_id=sc))

		return query_name, support_names, support_classes, query_has_mask

	def sample_sliding_window_episode(self, idx):
		r"""Sample episode using sliding window tile coordinates.
		
		Returns a tile-based query from query_set and samples support images
		from the support pools.
		"""
		if len(self.query_tiles) == 0:
			raise RuntimeError('Sliding window query tiles not built. Check query_set and sliding settings.')

		idx = idx % len(self.query_tiles)
		tile_info = self.query_tiles[idx]
		query_name = self.entry_image_name(tile_info)
		query_tile_coords = self.entry_coords(tile_info)
		
		query_has_mask = self.query_has_mask.get(query_name, True)
		
		# Sample support tiles from the support pools.
		support_names = []
		support_classes = []
		
		if self.split in ['val', 'test'] and self.eval_classes is not None:
			support_classes = list(self.active_class_ids)
		else:
			# Preserve the declared class order for deterministic evaluation.
			support_classes = list(self.class_ids[:self.way])
		
		def sample_from_pool(pool, query_name, shot):
			query_sig = self.entry_signature(query_name)
			candidates = [entry for entry in pool if self.entry_signature(entry) != query_sig]
			if not candidates:
				candidates = list(pool)
			if not candidates:
				raise RuntimeError('Empty support candidates while sampling sliding window episode')
			replace = len(candidates) < shot
			return np.random.choice(candidates, shot, replace=replace).tolist()
		
		def select_support(pool, query_name, shot, class_id=None):
			if self.support_strategy == 'random':
				return sample_from_pool(pool, query_name, shot)
			if self.support_strategy == 'max_area' and class_id is not None:
				return self.sample_max_area_support(pool, query_name, shot, class_id)
			if self.support_strategy == 'min_area' and class_id is not None:
				return self.sample_min_area_support(pool, query_name, shot, class_id)
			if self.support_strategy == 'similarity':
				return self.sample_similarity_support(pool, query_name, shot, tile_coords=query_tile_coords)
			return sample_from_pool(pool, query_name, shot)
		
		# If using a global similarity strategy, precompute a global ranked list
		# and split it into `way` chunks of `shot` each.
		global_similarity_chunks = None
		if self.support_strategy == 'similarity_global':
			total_needed = int(self.way) * int(self.shot)
			global_list = self.sample_similarity_global(None, query_name, total_needed, tile_coords=query_tile_coords)
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
		
		return query_name, support_names, support_classes, query_has_mask, query_tile_coords

	def load_sliding_window_pools(self, oem_val_pools_name, default_split=None):
		r"""Load sliding window pool tiles from JSON file.
		
		Expected format:
		{
			"class_id": [
				{"image_name": "...", "x": 0, "y": 0},
				...
			],
			...
		}
		"""
		pool_path = self.resolve_split_path(oem_val_pools_name, oem_val_pools_name)
		
		if not os.path.exists(pool_path):
			print(f'[Warning] Sliding window pool file not found: {pool_path}')
			return None
		
		try:
			with open(pool_path, 'r', encoding='utf-8') as f:
				pools_data = json.load(f)
			
			allowed_classes = set(self.active_class_ids if self.eval_classes is not None else self.class_ids)
			# Convert string keys to integers
			pools = {}
			for class_id_str, tiles in pools_data.items():
				class_id = int(class_id_str)
				# Filter to the active eval class prefix when requested.
				if class_id in allowed_classes:
					normalized_tiles = []
					for tile in tiles:
						if isinstance(tile, dict):
							tile_entry = dict(tile)
							tile_entry.setdefault('split', default_split)
							normalized_tiles.append(tile_entry)
						else:
							normalized_tiles.append(tile)
					pools[class_id] = normalized_tiles
			
			if not pools:
				print(f'[Warning] No matching classes in sliding window pools after filtering')
				return None
			
			total_tiles = sum(len(tiles) for tiles in pools.values())
			print(f'[{self.split}] Loaded sliding window pools with {total_tiles} tiles')
			return pools
		except Exception as e:
			print(f'[Error] Failed to load sliding window pools: {e}')
			return None

	def build_class_ids(self, eval_classes=None):
		# OEM-GFSS style split for normal FSS training/evaluation.
		base_class_ids = [1, 2, 3, 4, 5, 6, 7]
		novel_class_ids = [8, 9, 10, 11]
		if eval_classes is not None:
			class_ids = list(eval_classes)
		else:
			if self.split == 'trn':
				class_ids = base_class_ids
			else:
				class_ids = novel_class_ids

		if len(class_ids) == 0:
			raise ValueError(f'No class ids available for split={self.split}')

		return class_ids

	def build_pools(self):
		support_pool = {i: [] for i in self.class_ids}
		support_pool[21] = []
		support_pool[30] = []
		query_pool = []

		if self.use_sliding_window and self.split in ['val', 'test'] and self.support_sliding_window_pools is not None:
			if self.support_sliding_window_pools is None:
				raise RuntimeError('OEM sliding-window eval requires support pool tiles (train pools)')

			active_ids = self.active_class_ids
			for class_id in active_ids:
				support_pool[class_id] = list(self.support_sliding_window_pools.get(class_id, []))

			# Query pool comes from val.json query_set / test.json query_set, not from class pools.
			json_path = self.val_json_path if self.split == 'val' else self.test_json_path
			with open(json_path, 'r', encoding='utf-8') as f:
				split_data = json.load(f)
			for name in split_data.get('query_set', []):
				entry = os.path.join('valset' if self.split == 'val' else 'testset', 'labels', name).replace('\\', '/')
				if os.path.exists(os.path.join(self.ann_path, entry)):
					query_pool.append(entry)
					self.query_has_mask[entry] = True
			for class_id in active_ids:
				support_pool[class_id] = self.dedupe_entries(support_pool[class_id])

			# Provide combined pools for special strategies.
			support_pool[21] = [entry for class_id in active_ids for entry in support_pool[class_id]]
			support_pool[30] = list(support_pool[21])

			query_with_mask = len(query_pool)
			print(f'[{self.split}] query entries: {len(query_pool):,} (with label: {query_with_mask:,}, without label: 0)')
			support_counts = ', '.join([f'{class_id}:{len(support_pool[class_id])}' for class_id in active_ids])
			print(f'[{self.split}] support entries per class: {support_counts}')
			self.validate_split_purity(query_pool, support_pool)
			return query_pool, support_pool

		if self.split == 'trn':
			with open(self.train_list_path, 'r', encoding='utf-8') as f:
				train_names = [line.strip() for line in f if line.strip()]

			for name in train_names:
				entry = os.path.join('trainset', 'labels', name).replace('\\', '/')
				if not os.path.exists(os.path.join(self.ann_path, entry)):
					continue
				query_pool.append(entry)
				self.query_has_mask[entry] = True

				mask = self.read_mask(entry)
				classes_present = set(torch.unique(mask).tolist())
				for class_id in self.class_ids:
					if class_id in classes_present:
						support_pool[class_id].append(entry)

		else:
			split_dir = 'valset' if self.split == 'val' else 'testset'
			json_path = self.val_json_path if self.split == 'val' else self.test_json_path
			with open(json_path, 'r', encoding='utf-8') as f:
				split_data = json.load(f)

			support_json = split_data.get('support_set', {})
			for class_id in self.class_ids:
				names = list(support_json.get(str(class_id), [])) + list(support_json.get(class_id, []))
				entries = []
				for name in names:
					entry = os.path.join(split_dir, 'labels', name).replace('\\', '/')
					if os.path.exists(os.path.join(self.ann_path, entry)):
						entries.append(entry)
						self.query_has_mask[entry] = True
				support_pool[class_id] = entries

			if self.split == 'val':
				for name in split_data.get('query_set', []):
					entry = os.path.join(split_dir, 'labels', name).replace('\\', '/')
					if os.path.exists(os.path.join(self.ann_path, entry)):
						query_pool.append(entry)
						self.query_has_mask[entry] = True
				if len(query_pool) == 0:
					raise RuntimeError('OEM val split has no valid query labels. Check --oem_val_json and valset/labels.')
			else:
				for name in split_data.get('query_set', []):
					entry = os.path.join(split_dir, 'labels', name).replace('\\', '/')
					query_pool.append(entry)
					self.query_has_mask[entry] = os.path.exists(os.path.join(self.ann_path, entry))
				if len(query_pool) == 0:
					raise RuntimeError('OEM test split has empty query_set. Check --oem_test_json.')

		query_pool = self.dedupe_entries(query_pool)
		for i in self.class_ids:
			support_pool[i] = self.dedupe_entries(support_pool[i])
			support_pool[21] += support_pool[i]
			if support_pool[i]:
				nsel = min(20, len(support_pool[i]))
				support_pool[30] += random.sample(support_pool[i], nsel)

		query_with_mask = sum(1 for entry in query_pool if self.query_has_mask.get(entry, False))
		query_without_mask = len(query_pool) - query_with_mask
		print(f'[{self.split}] query entries: {len(query_pool):,} (with label: {query_with_mask:,}, without label: {query_without_mask:,})')
		if self.split in ['val', 'test']:
			support_counts = ', '.join([f'{class_id}:{len(support_pool[class_id])}' for class_id in self.class_ids])
			print(f'[{self.split}] support entries per class: {support_counts}')

		self.validate_split_purity(query_pool, support_pool)
		return query_pool, support_pool

	def build_img_metadata_classwise(self):
		img_metadata_classwise = {}
		class_ids = self.class_ids if self.split == 'trn' or self.eval_classes is None else self.active_class_ids
		for class_id in class_ids:
			if self.use_sliding_window and self.split in ['val', 'test']:
				candidates = list(self.support_pool[class_id])
			elif self.split in ['trn', 'val']:
				candidates = list(self.support_pool[class_id])
			else:
				# test.json query_set has no labels; evaluate each query against each novel class.
				candidates = list(self.query_pool)
				if len(candidates) == 0:
					candidates = list(self.support_pool[class_id])

			if len(candidates) == 0:
				raise RuntimeError(f'No metadata found for class {class_id} in split {self.split}')

			img_metadata_classwise[class_id] = candidates

		return img_metadata_classwise

	def build_img_metadata(self):
		img_metadata = []
		for class_id in self.active_class_ids:
			for img_name in self.img_metadata_classwise[class_id]:
				entry_name = self.entry_image_name(img_name)
				img_metadata.append([entry_name, class_id, bool(self.query_has_mask.get(entry_name, True))])

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

	def resolve_split_path(self, raw_path, default_name):
		if raw_path in ['', None]:
			raw_path = default_name
		raw_path = os.path.expanduser(raw_path)
		if os.path.isabs(raw_path):
			return raw_path
		candidate_paths = [
			raw_path,
			os.path.join(os.path.dirname(__file__), raw_path),
			os.path.join(self.img_path, raw_path),
			os.path.join(self.pool_path, raw_path),
		]
		for candidate in candidate_paths:
			if os.path.exists(candidate):
				return candidate
		return candidate_paths[0]

	def validate_split_purity(self, query_pool, support_pool):
		if self.split not in ['val', 'test']:
			return
		expected_prefix = 'valset/' if self.split == 'val' else 'testset/'
		allowed_support_prefixes = {'trainset/', expected_prefix}
		for entry in query_pool:
			entry_name = self.entry_image_name(entry)
			if not entry_name.startswith(expected_prefix):
				raise RuntimeError(f'Query entry out of split {self.split}: {entry}')
		for class_id in self.class_ids:
			for entry in support_pool[class_id]:
				entry_name = self.entry_image_name(entry)
				if not any(entry_name.startswith(prefix) for prefix in allowed_support_prefixes):
					raise RuntimeError(f'Support entry out of split {self.split}: {entry}')

	def sample_max_area_support(self, pool, query_name, shot, class_id):
		query_sig = self.entry_signature(query_name)
		candidates = [entry for entry in pool if self.entry_signature(entry) != query_sig]
		if not candidates:
			candidates = list(pool)
		if not candidates:
			raise RuntimeError('Empty support candidates while sampling max_area support')

		if any(isinstance(entry, dict) for entry in candidates):
			ranked = sorted(candidates, key=lambda entry: self.entry_class_fraction(entry, class_id), reverse=True)
		else:
			class_scores = self.class_representativity.get(class_id, {}) if self.class_representativity else {}
			ranked = sorted(candidates, key=lambda name: class_scores.get(name, 0.0), reverse=True)

		if len(ranked) >= shot:
			return ranked[:shot]
		if len(ranked) == 1:
			return ranked * shot
		return ranked + np.random.choice(ranked, shot - len(ranked), replace=True).tolist()

	def sample_min_area_support(self, pool, query_name, shot, class_id):
		query_sig = self.entry_signature(query_name)
		candidates = [entry for entry in pool if self.entry_signature(entry) != query_sig]
		if not candidates:
			candidates = list(pool)
		if not candidates:
			raise RuntimeError('Empty support candidates while sampling min_area support')

		if any(isinstance(entry, dict) for entry in candidates):
			ranked = sorted(candidates, key=lambda entry: self.entry_class_fraction(entry, class_id))
		else:
			class_scores = self.class_representativity.get(class_id, {}) if self.class_representativity else {}
			ranked = sorted(candidates, key=lambda name: class_scores.get(name, 0.0))

		if len(ranked) >= shot:
			return ranked[:shot]
		if len(ranked) == 1:
			return ranked * shot
		return ranked + np.random.choice(ranked, shot - len(ranked), replace=True).tolist()

	def sample_similarity_support(self, pool, query_name, shot, tile_coords=None):
		query_sig = self.entry_signature(query_name)
		candidates = [entry for entry in pool if self.entry_signature(entry) != query_sig]
		if not candidates:
			candidates = list(pool)
		if not candidates:
			raise RuntimeError('Empty support candidates while sampling similarity support')

		if self.similarity_topk is not None:
			lookup_key = query_sig
			if tile_coords is not None and isinstance(query_name, str):
				lookup_key = self.entry_signature({
					'split': 'valset' if self.split == 'val' else 'testset',
					'image_name': os.path.basename(query_name),
					'x': int(tile_coords['x']),
					'y': int(tile_coords['y']),
					'size': int(tile_coords['size']),
				})
			cached_topk = self.similarity_topk.get(lookup_key)
			if cached_topk:
				ordered = [entry for entry in cached_topk if self.entry_signature(entry) != query_sig]
				if not ordered:
					ordered = cached_topk
				if len(ordered) >= shot:
					random.shuffle(ordered)
					return ordered[:shot]
				if len(ordered) == 1:
					return ordered * shot
				if len(ordered) > 1:
					random.shuffle(ordered)
					return ordered + np.random.choice(ordered, shot - len(ordered), replace=True).tolist()

		if self.similarity_index is None or self.similarity_features is None:
			raise RuntimeError('Similarity strategy requires either a top-k JSON cache or a similarity NPZ cache')

		query_feat = self.get_query_feature(query_name, tile_coords=tile_coords)

	def sample_similarity_global(self, pool, query_name, total, tile_coords=None):
		"""Return a global ranked list of `total` support entries for `query_name`.
		This ignores per-class pools and uses the precomputed top-k JSON if available.
		Note: entries are returned as full entry objects (dict or string) and may belong
		to any class. Caller is responsible for splitting/assigning them to classes.
		"""
		query_sig = self.entry_signature(query_name)
		if self.similarity_topk is not None:
			lookup_key = query_sig
			if tile_coords is not None and isinstance(query_name, str):
				lookup_key = self.entry_signature({
					'split': 'valset' if self.split == 'val' else 'testset',
					'image_name': os.path.basename(query_name),
					'x': int(tile_coords['x']),
					'y': int(tile_coords['y']),
					'size': int(tile_coords['size']),
				})
			cached_topk = self.similarity_topk.get(lookup_key)
			if cached_topk:
				ordered = [entry for entry in cached_topk if self.entry_signature(entry) != query_sig]
				if not ordered:
					ordered = cached_topk
				# If not enough entries, pad with random choices from ordered
				if len(ordered) >= total:
					return ordered[:total]
				if len(ordered) == 0:
					raise RuntimeError('No similarity candidates available in topk cache')
				return ordered + np.random.choice(ordered, total - len(ordered), replace=True).tolist()

		# Fall back: require precomputed similarity index/features for full ranking
		raise RuntimeError('Global similarity sampling requires a top-k JSON cache (support_similarity_topk_cache)')
		candidate_pairs = []
		missing_entries = []
		for name in candidates:
			sig = self.entry_signature(name)
			if sig in self.similarity_index:
				candidate_pairs.append((name, self.similarity_features[self.similarity_index[sig]]))
			else:
				missing_entries.append(name)

		if missing_entries:
			for name in missing_entries:
				candidate_rgb = self.entry_similarity_rgb(name)
				candidate_feat = self.extract_similarity_feature(candidate_rgb)
				candidate_pairs.append((name, candidate_feat))

		if len(candidate_pairs) == 0:
			replace = len(candidates) < shot
			return np.random.choice(candidates, shot, replace=replace).tolist()

		candidate_feats = np.stack([feat for _, feat in candidate_pairs], axis=0)

		sims = np.dot(candidate_feats, query_feat)
		ranked_pos = np.argsort(-sims)
		ranked_names = [candidate_pairs[i][0] for i in ranked_pos]
		top_k = min(5, len(ranked_names))
		top_candidates = ranked_names[:top_k]
		if top_k > 0:
			random.shuffle(top_candidates)

		if len(top_candidates) >= shot:
			return top_candidates[:shot]
		if len(ranked_names) == 0:
			replace = len(candidates) < shot
			return np.random.choice(candidates, shot, replace=replace).tolist()
		if len(top_candidates) == 1:
			return top_candidates * shot
		if len(top_candidates) == 0:
			replace = len(candidates) < shot
			return np.random.choice(candidates, shot, replace=replace).tolist()
		return top_candidates + np.random.choice(top_candidates, shot - len(top_candidates), replace=True).tolist()

	def get_query_feature(self, query_name, tile_coords=None):
		if tile_coords is None:
			cache_key = query_name
			if cache_key not in self.query_feature_cache:
				query_img = self.read_img_array(query_name)
				self.query_feature_cache[cache_key] = self.extract_similarity_feature(query_img)
			return self.query_feature_cache[cache_key]

		tile_entry = {
			'split': 'valset' if self.split == 'val' else 'testset',
			'image_name': os.path.basename(query_name),
			'x': int(tile_coords['x']),
			'y': int(tile_coords['y']),
			'size': int(tile_coords['size']),
		}
		cache_key = self.entry_signature(tile_entry)
		if cache_key not in self.query_feature_cache:
			query_tile_rgb = self.entry_similarity_rgb(tile_entry)
			self.query_feature_cache[cache_key] = self.extract_similarity_feature(query_tile_rgb)
		return self.query_feature_cache[cache_key]

	def load_similarity_topk(self, cache_path):
		if not os.path.exists(cache_path):
			return None
		try:
			with open(cache_path, 'r', encoding='utf-8') as f:
				raw = json.load(f)
			topk = {}
			for query_sig, items in raw.items():
				entries = []
				for item in items:
					if isinstance(item, dict):
						support_sig = item.get('support')
					else:
						support_sig = item
					if support_sig is None:
						continue
					entries.append(self._support_signature_to_entry(support_sig))
				if entries:
					topk[query_sig] = entries
			return topk if topk else None
		except Exception as exc:
			print(f'[Warning] Failed to load similarity top-k cache {cache_path}: {exc}')
			return None

	def _support_signature_to_entry(self, signature):
		if not isinstance(signature, str) or '::' not in signature:
			return signature
		parts = signature.split('::')
		if len(parts) < 5:
			return signature
		split_dir, image_name = parts[0], parts[1]
		coords = {}
		for part in parts[2:]:
			if '=' in part:
				key, value = part.split('=', 1)
				coords[key] = int(value)
		entry = {'split': split_dir, 'image_name': image_name}
		entry.update(coords)
		return entry

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

		entries_by_key = {}
		for cls in self.class_ids:
			for entry in self.support_pool.get(cls, []):
				entries_by_key.setdefault(self.entry_signature(entry), entry)

		class_repr = {cls: {} for cls in self.class_ids}
		for entry in entries_by_key.values():
			mask = self.read_mask(self.entry_mask_path(entry)).numpy()
			total = float(mask.size)
			if total == 0:
				continue
			for cls in self.class_ids:
				class_repr[cls][self.entry_signature(entry)] = float((mask == cls).sum()) / total

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

		entries_by_key = {}
		for entry in list(self.query_pool):
			entries_by_key.setdefault(self.entry_signature(entry), entry)
		for cls in self.class_ids:
			for entry in self.support_pool.get(cls, []):
				entries_by_key.setdefault(self.entry_signature(entry), entry)
		entry_list = [entries_by_key[key] for key in sorted(entries_by_key.keys())]

		features = []
		for entry in entry_list:
			rgb = self.entry_similarity_rgb(entry)
			features.append(self.extract_similarity_feature(rgb))
		features = np.stack(features, axis=0).astype(np.float32)

		os.makedirs(os.path.dirname(cache_path), exist_ok=True)
		cache_entries = np.array(sorted(entries_by_key.keys()))
		np.savez_compressed(cache_path, entries=cache_entries, features=features)
		index = {name: i for i, name in enumerate(cache_entries.tolist())}
		return entry_list, features, index
