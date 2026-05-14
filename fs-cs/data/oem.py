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
		support_similarity_size=32,
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

		self.class_ids = self.build_class_ids()
		if self.way > len(self.class_ids):
			raise ValueError(f'way={self.way} is larger than available classes={len(self.class_ids)} for split={self.split}')

		self.query_pool, self.support_pool = self.build_pools()
		self.img_metadata_classwise = self.build_img_metadata_classwise()
		self.img_metadata = self.build_img_metadata()
		self.rdn_sup = rdn_sup

		self.support_strategy = support_strategy
		valid_strategies = {'random', 'max_area', 'similarity'}
		if self.support_strategy not in valid_strategies:
			raise ValueError(f'Invalid support_strategy={self.support_strategy}. Choices: {sorted(valid_strategies)}')

		self.class_representativity = None
		self.similarity_entries = None
		self.similarity_features = None
		self.similarity_index = None
		self.query_feature_cache = {}
		self.similarity_size = int(support_similarity_size)

		if self.split in ['val', 'test'] and self.support_strategy == 'max_area':
			area_cache_path = self.resolve_cache_path(support_area_cache, f'oem_class_representativity_{self.split}.json')
			self.class_representativity = self.load_or_build_class_representativity(area_cache_path)

		if self.split in ['val', 'test'] and self.support_strategy == 'similarity':
			sim_cache_path = self.resolve_cache_path(support_similarity_cache, f'oem_similarity_index_{self.split}.npz')
			(self.similarity_entries,
			 self.similarity_features,
			 self.similarity_index) = self.load_or_build_similarity_index(sim_cache_path)

	def __len__(self):
		if self.split == 'trn':
			return len(self.img_metadata)
		if self.split == 'val':
			return max(1000, len(self.img_metadata))
		return len(self.img_metadata)

	def __getitem__(self, idx):
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
		if self.split in ['val', 'test']:
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

	def load_frame(self, query_name, support_names, query_has_mask):
		query_img = self.read_img(query_name)
		query_mask = self.read_mask(query_name, allow_missing=not query_has_mask)
		if query_mask is None:
			query_mask = torch.zeros((query_img.size[1], query_img.size[0]), dtype=torch.long)
		support_imgs = [[self.read_img(name) for name in support_names_c] for support_names_c in support_names]
		support_masks = [[self.read_mask(name) for name in support_names_c] for support_names_c in support_names]

		org_qry_imsize = query_img.size

		return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

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

	def sample_episode(self, idx):
		if self.split in ['val', 'test']:
			np.random.seed(idx)

		idx %= len(self.img_metadata)
		query_name, query_class, query_has_mask = self.img_metadata[idx]

		support_names = []
		if self.way == 1:
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
			return sample_from_pool(pool, query_name, shot)

		if self.rdn_sup:
			for _ in support_classes:
				support_names.append(select_support(self.support_pool[21], query_name, self.shot, class_id=None))
		else:
			for sc in support_classes:
				if len(self.support_pool[sc]) == 0:
					raise RuntimeError(f'Empty support pool for class {sc} in {self.pool_path}')
				support_names.append(select_support(self.support_pool[sc], query_name, self.shot, class_id=sc))

		return query_name, support_names, support_classes, query_has_mask

	def build_class_ids(self):
		# OEM-GFSS style split for normal FSS training/evaluation.
		base_class_ids = [1, 2, 3, 4, 5, 6, 7]
		novel_class_ids = [8, 9, 10, 11]
		if self.split == 'trn':
			return base_class_ids
		return novel_class_ids

	def build_pools(self):
		support_pool = {i: [] for i in self.class_ids}
		support_pool[21] = []
		support_pool[30] = []
		query_pool = []

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

		query_pool = list(dict.fromkeys(query_pool))
		for i in self.class_ids:
			support_pool[i] = list(dict.fromkeys(support_pool[i]))
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
		for class_id in self.class_ids:
			if self.split in ['trn', 'val']:
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
		for class_id in self.class_ids:
			for img_name in self.img_metadata_classwise[class_id]:
				img_metadata.append([img_name, class_id, bool(self.query_has_mask.get(img_name, True))])

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
		return os.path.join(self.img_path, raw_path)

	def validate_split_purity(self, query_pool, support_pool):
		if self.split not in ['val', 'test']:
			return
		expected_prefix = 'valset/' if self.split == 'val' else 'testset/'
		for entry in query_pool:
			if not entry.startswith(expected_prefix):
				raise RuntimeError(f'Query entry out of split {self.split}: {entry}')
		for class_id in self.class_ids:
			for entry in support_pool[class_id]:
				if not entry.startswith(expected_prefix):
					raise RuntimeError(f'Support entry out of split {self.split}: {entry}')

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
