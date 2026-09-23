"""
Builds the 512x512 tile dataset (+ train/val/test CSVs) used to train the TIGER ROI classifier
(`training_TIGER_classifier.ipynb`), from a raw TIGER challenge download.

Assumes you have already downloaded the TIGER training data (tiger.grand-challenge.org, released
under CC BY-NC 4.0) and have the "tissue-bcss" ROI folder locally, with an `images/` subfolder of
ROI PNGs and a `masks/` subfolder of same-named, single-channel tissue-compartment masks (pixel
values 0-7):

    0 = exclude, 1 = invasive tumor, 2 = tumor-associated stroma, 3 = in-situ tumor,
    4 = healthy glands, 5 = necrosis (not in-situ), 6 = inflamed stroma, 7 = rest

For each ROI, this script:
  1. Slides a 512x512 window across the image/mask pair (50% overlap by default; a ROI smaller
     than 512 in either dimension is resized up instead of tiled).
  2. Labels each tile by the majority class among its mask pixels (excluding 0/"exclude"),
     skipping tiles where no class reaches --min_class_fraction of valid pixels (ambiguous/
     boundary tiles).
  3. Remaps the majority BCSS code to the 5 training classes via the same mapping used by
     `training_TIGER_classifier.ipynb`: {1:0, 2:1, 3:0, 5:2, 6:3, 7:4} (i.e. invasive + in-situ
     tumor merged into class 0; class 4 "healthy glands" is dropped, matching the training code).
  4. Saves each kept tile as a PNG and writes a stratified train/val/test split (columns `data`,
     `label`) compatible with `training_TIGER_classifier.ipynb`'s `TigerTileDataset`. The
     validation split is used for checkpoint selection during training; the test split is held
     out and evaluated only once, after training, to report the final metrics.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

LABEL_DICT = {1: 0, 2: 1, 3: 0, 5: 2, 6: 3, 7: 4}
EXCLUDE_CODE = 0

parser = argparse.ArgumentParser(description='Extract 512x512 labeled tiles from raw TIGER tissue-bcss ROIs.')
parser.add_argument('--images_dir', type=str, default='/path/to/your/TIGER/wsirois/roi-level-annotations/tissue-bcss/images',
                     help='directory of raw TIGER ROI PNGs')
parser.add_argument('--masks_dir', type=str, default='/path/to/your/TIGER/wsirois/roi-level-annotations/tissue-bcss/masks',
                     help='directory of matching tissue-compartment mask PNGs (same filenames as images_dir, single-channel, values 0-7)')
parser.add_argument('--output_dir', type=str, default='./TIGER_labeled_tiles_512', help='where extracted tile PNGs are saved')
parser.add_argument('--out_train_csv', type=str, default='./train_df_tiger_updated_512.csv')
parser.add_argument('--out_val_csv', type=str, default='./val_df_tiger_updated_512.csv')
parser.add_argument('--out_test_csv', type=str, default='./test_df_tiger_updated_512.csv')
parser.add_argument('--patch_size', type=int, default=512)
parser.add_argument('--overlap', type=float, default=0.5, help='fractional overlap between adjacent tiles')
parser.add_argument('--min_class_fraction', type=float, default=0.5,
                     help='a tile is kept only if its majority class covers at least this fraction of valid (non-excluded) mask pixels')
parser.add_argument('--val_size', type=float, default=0.15, help='fraction of all tiles held out for validation (checkpoint selection)')
parser.add_argument('--test_size', type=float, default=0.15, help='fraction of all tiles held out for the final, one-time test evaluation')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


def get_patch_coordinates(image_shape, patch_size, overlap):
    H, W = image_shape
    if H < patch_size or W < patch_size:
        return [(0, 0)]

    stride = max(int(patch_size * (1 - overlap)), 1)
    nh = int(np.ceil((H - patch_size) / stride)) + 1
    nw = int(np.ceil((W - patch_size) / stride)) + 1

    coords = set()
    for i in range(nh):
        for j in range(nw):
            y = min(i * stride, H - patch_size)
            x = min(j * stride, W - patch_size)
            coords.add((y, x))
    return sorted(coords)


def extract_patches(image, mask, patch_size):
    H, W = image.shape[:2]
    if H < patch_size or W < patch_size:
        resized_image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
        return [resized_image], [resized_mask], [(0, 0)]

    coords = get_patch_coordinates((H, W), patch_size, args.overlap)
    patches = [image[y:y + patch_size, x:x + patch_size] for y, x in coords]
    masks = [mask[y:y + patch_size, x:x + patch_size] for y, x in coords]
    return patches, masks, coords


def majority_label(mask_patch, min_fraction):
    values, counts = np.unique(mask_patch, return_counts=True)
    valid = values != EXCLUDE_CODE
    values, counts = values[valid], counts[valid]
    if len(counts) == 0:
        return None

    total_valid = counts.sum()
    best_idx = np.argmax(counts)
    best_code, best_count = int(values[best_idx]), counts[best_idx]

    if best_count / total_valid < min_fraction:
        return None
    return LABEL_DICT.get(best_code)


def main():
    os.makedirs(args.output_dir, exist_ok=True)

    image_files = sorted(f for f in os.listdir(args.images_dir) if f.lower().endswith('.png'))
    print(f'Found {len(image_files)} ROI images in {args.images_dir}')

    records = []
    skipped_no_mask = []

    for fname in tqdm(image_files, desc='ROIs'):
        mask_path = os.path.join(args.masks_dir, fname)
        if not os.path.isfile(mask_path):
            skipped_no_mask.append(fname)
            continue

        image = cv2.imread(os.path.join(args.images_dir, fname), cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            skipped_no_mask.append(fname)
            continue

        stem = os.path.splitext(fname)[0]
        patches, mask_patches, coords = extract_patches(image, mask, args.patch_size)

        for (y, x), img_patch, mask_patch in zip(coords, patches, mask_patches):
            label = majority_label(mask_patch, args.min_class_fraction)
            if label is None:
                continue

            tile_name = f'{stem}_{y}_{x}.png'
            tile_path = os.path.join(args.output_dir, tile_name)
            cv2.imwrite(tile_path, img_patch)
            records.append({'data': os.path.abspath(tile_path), 'label': label})

    print(f'\nExtracted {len(records)} labeled tiles from {len(image_files)} ROIs '
          f'({len(skipped_no_mask)} ROIs skipped for missing/unreadable image or mask)')
    if skipped_no_mask:
        print('Skipped:', skipped_no_mask[:20], '...' if len(skipped_no_mask) > 20 else '')

    label_df = pd.DataFrame.from_records(records)

    # two-stage split: carve off (val + test) first, then split that remainder into val/test
    holdout_size = args.val_size + args.test_size
    train_df, holdout_df = train_test_split(
        label_df, test_size=holdout_size, stratify=label_df['label'], random_state=args.seed
    )
    val_df, test_df = train_test_split(
        holdout_df, test_size=args.test_size / holdout_size,
        stratify=holdout_df['label'], random_state=args.seed
    )

    train_df.reset_index(drop=True).to_csv(args.out_train_csv, index=False)
    val_df.reset_index(drop=True).to_csv(args.out_val_csv, index=False)
    test_df.reset_index(drop=True).to_csv(args.out_test_csv, index=False)

    print(f'\ntrain: {len(train_df)} tiles -> {args.out_train_csv}')
    print(f'val:   {len(val_df)} tiles -> {args.out_val_csv}')
    print(f'test:  {len(test_df)} tiles -> {args.out_test_csv}')
    print('label counts (train):', train_df['label'].value_counts().sort_index().to_dict())
    print('label counts (val):  ', val_df['label'].value_counts().sort_index().to_dict())
    print('label counts (test): ', test_df['label'].value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
