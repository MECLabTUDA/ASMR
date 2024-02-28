import numpy as np
import os
import tifffile as tiff
import zarr
from PIL import TiffImagePlugin
from PIL.Image import Image
import pandas as pd
from tqdm import tqdm


def generate_new_compressed_tiff(source, target):
    TiffImagePlugin.WRITE_LIBTIFF = True
    compression = 'tiff_lzw'

    scanners = os.listdir(source)
    for scanner in scanners:

        images = os.listdir(os.path.join(source, scanner))
        for image in images:
            masks = os.listdir(os.path.join(source, scanner, image))
            for mask in masks:
                with Image.open(os.path.join(source, scanner, image, mask)) as im:

                    target_file = os.path.join(target, scanner, image)

                    if not os.path.exists(target_file):
                        os.makedirs(target_file, exist_ok=True)

                    im.save(os.path.join(target, scanner, image, mask), compression=compression, save_all=True)
                    print(os.path.join(target_file, mask))


def get_tile(image, size, overlap):
    '''
    gets the tiles from one image
    image: str path to image
    size: tuple of dimensions
    overlap: float
    '''
    img_tiff = tiff.imread(image, aszarr=True)
    img_zarr = zarr.open(img_tiff, mode='r')

    if isinstance(img_zarr, zarr.hierarchy.Group):
        img_zarr = img_zarr['0']

    img_size = img_zarr.shape

    # stepsize
    stepsize_y = size[0] - int(size[0] * overlap)
    stepsize_x = size[1] - int(size[1] * overlap)

    tiles = []

    # Sliding parameters
    y = size[0]
    x = size[1]

    while y <= img_size[0]:
        x = size[1]
        while x < img_size[1]:
            tiles.append([x - size[1], x, y - size[0], y])

            x += stepsize_x
        y += stepsize_y

    return tiles


def parse_data(data, tile_file):
    tiles = pd.DataFrame(columns=['image', 'scanner', 'tiles'])

    for image in tqdm(data):
        for tile in data[image]['tiles']:
            tiles.loc[len(tiles)] = [image, data[image]['scanner'], tile]

    tiles.to_pickle(tile_file)
    return tiles


def get_tiles(metadata, tile_file, image_path):
    data = {}
    overlap = 0.2
    size = 800
    if os.path.isfile(tile_file):
        tiles = pd.read_pickle(tile_file)
        return tiles

    for index, row in metadata.iterrows():
        image = os.path.join(image_path, row['scanner'], row['image'])
        image_tile = get_tile(image=image, overlap=overlap, size=size)

        data[row['image']] = {'tiles': image_tile, 'scanner': row['scanner']}

    return parse_data(data)


def generate_masks(root_dir):
    scanners = os.listdir(root_dir)
    for scanner in scanners:
        images = os.listdir(os.path.join(root_dir, scanner))
        for image in images:
            generate_full_mask(os.path.join(root_dir, scanner, image))


def generate_full_mask(image_path):
    mask_map = {'G5_Mask': 5, 'G4_Mask': 4, 'G3_Mask': 3,
                'Normal_Mask': 2, 'Stroma_Mask': 1}

    if not os.path.isfile(os.path.join(image_path, 'segmentation_mask.tif')):
        masks = os.listdir(image_path)
        masks = [mask for mask in masks if mask[:-4] in mask_map]
        masks = sorted(masks, key=lambda x: mask_map[x[:-4]])

        img_tiff = tiff.imread(os.path.join(image_path, masks[0]), aszarr=True)
        img_zarr = zarr.open(img_tiff, mode="r")

        dim = img_zarr.shape

        seg_mask = np.zeros(dim)

        for mask in masks:
            img_tiff = tiff.imread(os.path.join(image_path, mask), aszarr=True)
            img_zarr = zarr.open(img_tiff, mode="r")
            seg_mask[img_zarr] = mask_map[mask[:-4]]

        tiff.imwrite(os.path.join(image_path, 'segmentation_mask.tif'), seg_mask.astype('int'), compression='lzw')

        print(f'Saved segmentation mask: {image_path}')
    else:
        print('File already exist')


def is_valid(img_path, tiles):
    try:
        img_tiff = tiff.imread(img_path, aszarr=True)
        img_zarr = zarr.open(img_tiff, mode="r")

        patch = img_zarr[tiles[0]:tiles[1], tiles[2]:tiles[3]]

        total_elems = patch.shape[0] * patch.shape[1]
        ignore_elems = np.count_nonzero(patch == 0)

        ratio = ignore_elems / total_elems

    except:
        return False

    if ratio < 0.9:
        return True
    return False


def filter_tiles(root_path, tiles):
    valid_tiles = tiles
    valid_tiles['valid'] = np.nan

    for index, row in valid_tiles.iterrows():
        img_path = os.path.join(root_path, row['scanner'], row['image'][:-5], 'segmentation_mask.tif')
        valid_tiles.at[index, 'valid'] = is_valid(img_path, row['tiles'])

    valid_tiles = valid_tiles[valid_tiles['valid'] == True][['image', 'scanner', 'tiles']]

    valid_tiles.to_pickle('/local/scratch/AGGC/AGGC2022_test/Subset3_Test_tiles/valid_tiles.pkl')

    return valid_tiles


def get_cmplt_images(root_path):
    res = []
    others = []
    scanners = os.listdir(root_path)
    for scanner in scanners:
        images = os.listdir(os.path.join(root_path, scanner))
        for image in images:
            if len(os.listdir(os.path.join(root_path, scanner, image))) == 6:
                res.append(image)
            else:
                others.append(image)
    return res, others
