"""
Extracts candidate tissue patch coordinates from raw whole-slide images (WSIs), producing the
per-slide HDF5 coordinate files consumed by `labeling_TIGER_inference.py` (its `--coords_dir`).

For each WSI, this script:
  1. Builds a tissue foreground mask on a downsampled thumbnail, in three stages:
     - Chromatic-artifact removal: on a CLAHE-contrast-enhanced copy of the thumbnail, three
       YUV-color-space range filters flag pen-marking/annotation-like and other strongly colored
       artifacts; their union is hole-filled, inverted, and applied to the thumbnail so that
       flagged pixels are zeroed out. If the raw thumbnail is itself low-saturation, low-contrast,
       or skewed toward low intensities (at least 2 of these 3 heuristics), it is first run through
       an ImageJ-"Auto"-style per-channel contrast stretch before this filtering. The resulting
       artifact-masked image is the input to the next stages.
     - Stain normalization (enabled by default, --no_stain_normalize to disable): tissue pixels
       (Otsu-thresholded grayscale, further restricted to brightness < 240 and saturation > 30)
       have their HSV hue circularly shifted and saturation/value z-score-rescaled to match a
       reference color distribution (TCGA_BRCA_REFERENCE_STATS, from 1,133 TCGA-BRCA slides).
     - Saturation-based mask: convert to HSV, median-filter (7x7) the saturation channel, and
       threshold it (fixed value, default 10 on a 0-255 scale) -- H&E-stained tissue is strongly
       colored, while glass background is pale/washed out, so this separates tissue from
       background. Morphological closing (4x4) fills small gaps.
     - Gray/background-artifact-removal mask: discard near-gray pixels (red, green and blue
       channel values all within a small tolerance of one another -- dust, pen marks, and glass
       edges tend to be closer to gray than the pink/purple of stained tissue) together with pure
       white/black background pixels; keep only contours above a minimum area, then apply
       morphological closing (5x5 elliptical kernel, 2 iterations).
     - The saturation-based and gray-artifact-removal masks are combined with a logical AND to
       produce the final tissue mask.
  2. Lays a non-overlapping grid of tile_size x tile_size candidate patches (in full-resolution,
     level-0 pixel coordinates) across the slide. A tile is kept only if (a) at least
     --min_tile_coverage of its area (checked on the downsampled mask) is foreground, (b) its
     center pixel is foreground, and (c) at least 2 of its 4 border edges contain any foreground
     pixel (this combination discards tiles that only clip a thin sliver of tissue at a corner).
  3. Groups kept tiles by which contiguous tissue contour they fall in, and drops contours
     contributing fewer than --min_tiles patches (tiny, likely-spurious tissue fragments).
  4. Saves the retained (x, y) level-0 pixel coordinates per slide to an HDF5 file with a
     `coords` dataset and a `metadata` group (tile_size, overlap, total_tiles) -- the same layout
     `labeling_TIGER_inference.py` expects.
"""
import argparse
import os

import cv2
import h5py
import numpy as np
import openslide
from tqdm import tqdm

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description='Extract candidate tissue patch coordinates from raw WSIs.')
parser.add_argument('--slide_dir', type=str, default='/path/to/your/wsi', help='directory of WSI files (.svs)')
parser.add_argument('--output_dir', type=str, default=os.path.join(REPO_DIR, 'TIGER_training', 'candidate_patch_coords'),
                     help='where per-slide .h5 coordinate files are saved')
parser.add_argument('--tile_size', type=int, default=512)
parser.add_argument('--overlap', type=float, default=0.0, help='fractional overlap between adjacent tiles')
parser.add_argument('--thumb_downscale', type=int, default=64,
                     help='downscale factor for the thumbnail the foreground mask is computed on')
parser.add_argument('--sat_threshold', type=int, default=10, help='fixed saturation threshold, 0-255 scale')
parser.add_argument('--gray_tolerance', type=int, default=15,
                     help='a pixel is treated as gray/non-tissue if its R, G, B values are all within this many levels of each other')
parser.add_argument('--min_area', type=int, default=10, help='minimum contour area (thumbnail scale) kept by the artifact-removal mask')
parser.add_argument('--min_tile_coverage', type=float, default=0.5,
                     help='a candidate tile is kept only if at least this fraction of its area (thumbnail scale) is foreground')
parser.add_argument('--min_tiles', type=int, default=5,
                     help='tissue contours contributing fewer than this many valid tiles are dropped entirely '
                          '(organ/cohort-dependent; the original framework used a larger value for some organs -- tune as needed)')
parser.add_argument('--min_hole_area', type=int, default=10,
                     help='connected background regions at most this many pixels (thumbnail scale) are filled in as artifact when hole-filling the chromatic-artifact mask')
parser.add_argument('--no_stain_normalize', dest='stain_normalize', action='store_false', default=True,
                     help='disable HSV-statistics stain normalization (to a TCGA-BRCA reference; see TCGA_BRCA_REFERENCE_STATS). '
                          'Enabled by default, matching the setting used to generate the coordinates in this study.')
args = parser.parse_args()

# HSV reference statistics for stain normalization, computed from 1,133 TCGA-BRCA slides
# (tissue pixels: Otsu-thresholded grayscale AND brightness<240 AND saturation>30).
TCGA_BRCA_REFERENCE_STATS = {
    'hue_mean': 155.609, 'hue_std': 7.053,
    'sat_mean': 77.807, 'sat_std': 18.849,
    'val_mean': 185.633, 'val_std': 18.607,
}


# ---- Optional stain normalization (off by default; see --stain_normalize) ----

def stain_tissue_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bright_enough = hsv[:, :, 2] < 240
    saturated_enough = hsv[:, :, 1] > 30
    return (otsu_mask.astype(bool)) & bright_enough & saturated_enough


def hsv_stain_stats(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    return {
        'hue_mean': hsv[mask, 0].mean(), 'hue_std': hsv[mask, 0].std(),
        'sat_mean': hsv[mask, 1].mean(), 'sat_std': hsv[mask, 1].std(),
        'val_mean': hsv[mask, 2].mean(), 'val_std': hsv[mask, 2].std(),
    }


def normalize_stain(image, target_stats):
    """Shifts hue (circularly) and z-score-rescales saturation/value of tissue pixels to match
    the target (TCGA-BRCA reference) HSV statistics."""
    mask = stain_tissue_mask(image)
    if not mask.any():
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    current = hsv_stain_stats(image, mask)

    hue_diff = target_stats['hue_mean'] - current['hue_mean']
    if abs(hue_diff) > 90:
        hue_diff += -180 if hue_diff > 0 else 180

    pixels = hsv[mask]
    pixels[:, 0] = (pixels[:, 0] + hue_diff) % 180
    if current['sat_std'] > 0:
        pixels[:, 1] = np.clip((pixels[:, 1] - current['sat_mean']) / current['sat_std'] * target_stats['sat_std'] + target_stats['sat_mean'], 0, 255)
    if current['val_std'] > 0:
        pixels[:, 2] = np.clip((pixels[:, 2] - current['val_mean']) / current['val_std'] * target_stats['val_std'] + target_stats['val_mean'], 0, 255)
    hsv[mask] = pixels

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ---- Stage 1: chromatic-artifact removal (pen marks etc.) ----

def imagej_auto_contrast(image):
    """Per-channel percentile-clip contrast stretch, matching ImageJ's 'Auto' brightness/contrast."""
    auto_threshold = 5000
    mins, maxs = [], []
    for c in range(image.shape[2]):
        hist, _ = np.histogram(image[:, :, c], bins=256, range=(0, 255))
        total = hist.sum()
        if total < auto_threshold:
            lo = int(np.argmax(hist > 0))
            hi = 255 - int(np.argmax(hist[::-1] > 0))
            lo, hi = max(0, lo - 1), min(255, hi + 1)
        else:
            thresh = int(total * 0.0001)
            cum = 0
            lo = 0
            for i in range(256):
                cum += hist[i]
                if cum > thresh:
                    lo = i
                    break
            cum = 0
            hi = 255
            for i in range(255, -1, -1):
                cum += hist[i]
                if cum > thresh:
                    hi = i
                    break
        mins.append(lo)
        maxs.append(hi)

    global_min, global_max = max(mins), min(maxs)
    out = np.zeros_like(image)
    for c in range(image.shape[2]):
        channel = image[:, :, c].astype(np.float32)
        if global_max > global_min:
            out[:, :, c] = np.clip((channel - global_min) * 255.0 / (global_max - global_min), 0, 255)
        else:
            out[:, :, c] = channel
    return out.astype(np.uint8)


def needs_contrast_adjustment(image):
    """Flags thumbnails that are low-saturation, low-contrast, or skewed dark -- these get an
    ImageJ-Auto-style contrast stretch before chromatic-artifact filtering (>=2 of 3 heuristics)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bright_enough = hsv[:, :, 2] / 255.0 < 0.9
    sat = hsv[:, :, 1] / 255.0
    mean_sat = sat[bright_enough].mean() if bright_enough.any() else sat.mean()
    low_saturation = mean_sat < 0.15

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    low_contrast = gray.std() < 25

    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    low_intensity_ratio = hist[:128].sum() / gray.size
    low_intensity = low_intensity_ratio > 0.65

    return sum([low_saturation, low_contrast, low_intensity]) >= 2


def clahe_enhance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def yuv_range_mask(image, lo, hi):
    yuv = cv2.cvtColor(clahe_enhance(image), cv2.COLOR_RGB2YUV)
    return cv2.inRange(yuv, lo, hi)


def cleanup(mask, kernel_size, open_close_only=False, dilate_iterations=0):
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if dilate_iterations:
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
    return mask


def fill_small_holes(mask, min_hole_area):
    inverted = cv2.bitwise_not(mask)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    filled = mask.copy()
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] <= min_hole_area:
            filled[labels == i] = 255
    return filled


def remove_chromatic_artifacts(image):
    """Zeroes out pen-marking/annotation-like and other strongly colored (YUV-flagged) pixels.
    Returns the artifact-masked image to be used as input for the saturation/gray-based masks."""
    if needs_contrast_adjustment(image):
        image = imagej_auto_contrast(image)

    red_mask = cleanup(yuv_range_mask(image, (0, 0, 225), (255, 120, 255)), (3, 3))
    green_mask = cleanup(yuv_range_mask(image, (0, 90, 0), (255, 255, 120)), (7, 7), dilate_iterations=2)
    blue_mask = cleanup(yuv_range_mask(image, (0, 130, 0), (255, 255, 120)), (7, 7), dilate_iterations=2)

    artifact_mask = fill_small_holes(red_mask | green_mask | blue_mask, args.min_hole_area)
    keep_mask = cv2.bitwise_not(artifact_mask)
    return cv2.bitwise_and(image, image, mask=keep_mask)


def get_thumbnail_and_scaler(slide):
    if len(slide.level_dimensions) > 1:
        level = -1
        thumbnail = np.array(slide.get_thumbnail(slide.level_dimensions[level]))
        scaler = int(slide.level_downsamples[level])
    else:
        w0, h0 = slide.level_dimensions[0]
        thumbnail = np.array(slide.get_thumbnail((w0 // args.thumb_downscale, h0 // args.thumb_downscale)))
        scaler = args.thumb_downscale
    return thumbnail[:, :, :3], scaler


def saturation_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    sat = cv2.medianBlur(hsv[:, :, 1], 7)
    _, mask = cv2.threshold(sat, args.sat_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((4, 4), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def artifact_removal_mask(image):
    rgb = image.astype(np.int16)
    rg = np.abs(rgb[:, :, 0] - rgb[:, :, 1]) <= args.gray_tolerance
    rb = np.abs(rgb[:, :, 0] - rgb[:, :, 2]) <= args.gray_tolerance
    gb = np.abs(rgb[:, :, 1] - rgb[:, :, 2]) <= args.gray_tolerance
    not_gray = ~(rg & rb & gb)

    g = image[:, :, 1]
    not_background = (g > 0) & (g < 255)

    mask = ((not_gray & not_background).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= args.min_area:
            cv2.drawContours(cleaned, [c], -1, 255, thickness=cv2.FILLED)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)


def tissue_mask(image):
    artifact_masked = remove_chromatic_artifacts(image)
    if args.stain_normalize:
        artifact_masked = normalize_stain(artifact_masked, TCGA_BRCA_REFERENCE_STATS)
    return cv2.bitwise_and(saturation_mask(artifact_masked), artifact_removal_mask(artifact_masked))


def is_tile_valid(tile_mask):
    th, tw = tile_mask.shape
    if tile_mask[th // 2, tw // 2] == 0:
        return False
    edge_count = (np.count_nonzero(tile_mask[0, :]) + np.count_nonzero(tile_mask[-1, :]) +
                  np.count_nonzero(tile_mask[:, 0]) + np.count_nonzero(tile_mask[:, -1]))
    return edge_count >= 2


def sample_candidate_tiles(mask, level0_width, level0_height, scaler):
    binary_mask = (mask > 0).astype(np.uint8)
    mask_tile_size = round(args.tile_size / scaler)
    stride = int(args.tile_size - args.tile_size * args.overlap)

    coords = []
    for y in range(0, level0_height, stride):
        for x in range(0, level0_width, stride):
            y_idx, x_idx = round(y / scaler), round(x / scaler)
            tile = binary_mask[y_idx:y_idx + mask_tile_size, x_idx:x_idx + mask_tile_size]
            if tile.shape[0] != mask_tile_size or tile.shape[1] != mask_tile_size:
                continue
            if np.sum(tile) < (mask_tile_size ** 2) * args.min_tile_coverage:
                continue
            if is_tile_valid(tile):
                coords.append((x, y))
    return coords, binary_mask, mask_tile_size


def drop_small_tissue_fragments(coords, binary_mask, mask_tile_size, scaler):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    clusters = {}
    for x, y in coords:
        x_idx, y_idx = round(x / scaler), round(y / scaler)
        centroid = (x_idx + mask_tile_size / 2, y_idx + mask_tile_size / 2)
        for idx, contour in enumerate(contours):
            if cv2.pointPolygonTest(contour, centroid, False) >= 0:
                clusters.setdefault(idx, []).append((x, y))
                break

    kept = []
    for cluster in clusters.values():
        if len(cluster) >= args.min_tiles:
            kept.extend(cluster)
    return kept


def save_hdf5(out_path, coords):
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('coords', data=np.array(coords), compression='gzip')
        meta = hf.create_group('metadata')
        meta.attrs['tile_size'] = args.tile_size
        meta.attrs['overlap'] = args.overlap
        meta.attrs['total_tiles'] = len(coords)


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    slide_files = sorted(f for f in os.listdir(args.slide_dir) if f.lower().endswith('.svs'))
    print(f'Found {len(slide_files)} slides in {args.slide_dir}')

    for fname in tqdm(slide_files, desc='Slides'):
        sample_name = os.path.splitext(fname)[0]
        out_path = os.path.join(args.output_dir, f'{sample_name}.h5')
        if os.path.isfile(out_path):
            continue  # already processed, resume-safe

        try:
            slide = openslide.OpenSlide(os.path.join(args.slide_dir, fname))
            level0_width, level0_height = slide.level_dimensions[0]
            thumbnail, scaler = get_thumbnail_and_scaler(slide)
            slide.close()

            mask = tissue_mask(thumbnail)
            coords, binary_mask, mask_tile_size = sample_candidate_tiles(mask, level0_width, level0_height, scaler)
            coords = drop_small_tissue_fragments(coords, binary_mask, mask_tile_size, scaler)

            save_hdf5(out_path, coords)
        except Exception as e:
            print(f'FAILED {sample_name}: {e}')

    print(f'\nDone. Coordinate files written to {args.output_dir}')


if __name__ == '__main__':
    main()
