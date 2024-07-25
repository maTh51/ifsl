import os
import re
from itertools import product
import rasterio as rio
from rasterio import windows
import numpy as np
from collections import defaultdict

def get_tiles(ds, width, height):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off, row_off=row_off,
                                width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

def process(in_path, in_filename, out_path, out_filename, tile_width, tile_height, sector=''):
    with rio.open(os.path.join(in_path, in_filename)) as inds:
        meta = inds.meta.copy()
        for window, transform in get_tiles(inds, tile_width, tile_height):
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            if meta['width'] == 400 and meta['height'] == 400:
                out_filename_with_sector = f"{out_filename}_{sector}_{int(window.col_off)}_{int(window.row_off)}.tif"
                outpath = os.path.join(out_path, out_filename_with_sector)
                with rio.open(outpath, 'w', **meta) as outds:
                    outds.write(inds.read(window=window))

def process_files(in_path, out_path, filenames, filename_pattern, tile_width=400, tile_height=400, is_mask=False):
    for filename in filenames:
        # Extrair o setor e ID da imagem do nome do arquivo
        match = re.match(r'(m_\d+)_([a-z]+)_.*', filename)
        if match:
            id_part, sector = match.groups()
            out_filename = filename_pattern.format(id_part)
            if is_mask:
                process(in_path, filename, out_path + "/masks", out_filename, tile_width, tile_height, sector)
                count_classes(os.path.join(in_path, filename), out_path + "/pools", out_filename, tile_width, tile_height, sector)
            else:
                process(in_path, filename, out_path + "/images", out_filename, tile_width, tile_height, sector)

def count_classes(filepath, out_path, out_filename, tile_width, tile_height, sector):
    with rio.open(filepath) as src:
        for window, _ in get_tiles(src, tile_width, tile_height):
            mask_data = src.read(1, window=window)
            unique, counts = np.unique(mask_data, return_counts=True)
            total_pixels = mask_data.size
            class_counts = dict(zip(unique, counts))
            for cls, count in class_counts.items():
                if count / total_pixels >= 0.10:  # Pelo menos 10% da classe
                    class_file = os.path.join(out_path, f'{cls}.txt')
                    with open(class_file, 'a') as f:
                        tile_name = f"{out_filename}_{sector}_{int(window.col_off)}_{int(window.row_off)}.tif"
                        f.write(os.path.join(out_path, tile_name) + '\n')

def process_directory(in_path, out_path, directory_pattern, is_mask=False):
    directory = os.listdir(in_path)
    name_dir_mask = [aux for aux in directory if re.search(directory_pattern, aux)]
    process_files(in_path, out_path, name_dir_mask, '{}', is_mask=is_mask)

if __name__ == "__main__":
    in_path = "/scratch/dataset/chesapeake/ny_1m_2013_extended-debuffered-train_tiles"
    out_path = "/scratch/matheuspimenta/chesapeake_400"
    dataset = in_path.split("/")[-1].split("_")[0]

    # Images
    # directory_pattern_image = r'm_\d+_[a-z]+_\d+_\d+_naip-new'
    # process_directory(in_path, out_path, directory_pattern_image)

    # Masks
    directory_pattern_mask = r'm_\d+_[a-z]+_\d+_\d+_lc'
    process_directory(in_path, out_path, directory_pattern_mask, is_mask=True)
