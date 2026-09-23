import os
import numpy as np
import pandas as pd
import torch
import openslide
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch.nn as nn
from tqdm import tqdm
import h5py

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import StudentModel_convnext, Student_Projection_Head, Student_convnext_backbone, Patch_Classifier_Softmax_KD

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import random
from collections import defaultdict

import argparse
mpl.rcParams["figure.dpi"] = 300

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_EXTRACTOR_DIR = os.path.join(REPO_DIR, 'G2B_BRCA.pth')
TIGER_MODEL_PATH = os.path.join(REPO_DIR, 'TIGER_training', 'tiger_weights', 'best_model.pth')

parser = argparse.ArgumentParser(description='Run the trained TIGER ROI classifier over your WSI cohort, keeping tumor/TAS/necrosis/inflamed-stroma patches.')
parser.add_argument('--coords_dir', type=str, default=os.path.join(REPO_DIR, 'TIGER_training', 'candidate_patch_coords'),
                     help='directory of per-slide .h5 files with a "coords" dataset of candidate tissue-tile (x, y) coordinates '
                          '(pre-ROI-filtering; produced by extract_candidate_patches.py)')
parser.add_argument('--slide_dir', type=str, default='/path/to/your/wsi', help='directory of WSI files (.svs)')
parser.add_argument('--clinical_dir', type=str, default='/path/to/your/clinical_info.csv', help='clinical CSV (needs at least a tube label column)')
parser.add_argument('--output_dir', type=str,
                     default=os.path.join(REPO_DIR, 'TIGER_training', 'ROI_sampling_all'))
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=512)
parser.add_argument('--num_vis_samples', type=int, default=15)
parser.add_argument('--num_workers', type=int, default=16)
args = parser.parse_args()

DEVICE = args.device


def sample_patches_by_class(roi_coords, prediction_results, num_samples=10, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)

    class_coords = defaultdict(list)
    for coord, pred in zip(roi_coords, prediction_results):
        class_coords[pred].append(tuple(coord))

    sampled_data = {}
    for class_name, coords_list in class_coords.items():
        if len(coords_list) >= num_samples:
            sampled_coords = random.sample(coords_list, num_samples)
        else:
            sampled_coords = coords_list
        sampled_data[class_name] = sampled_coords

    return sampled_data


def visualize_patches_grid(slide, sampled_data, patch_size=512,
                            max_cols=10, figsize=(20, 12), save_path=None):
    class_name_mapping = {
        0: 'Tumor',
        1: 'Tumor-associated Stroma',
        2: 'Necrosis',
        3: 'Inflamed Stroma',
        4: 'Rest'
    }

    sorted_class_ids = sorted(sampled_data.keys())
    num_classes = len(sorted_class_ids)
    total_rows = num_classes

    fig, axes = plt.subplots(total_rows, max_cols, figsize=figsize)
    if total_rows == 1:
        axes = axes.reshape(1, -1)
    if max_cols == 1:
        axes = axes.reshape(-1, 1)

    for class_idx, class_id in enumerate(sorted_class_ids):
        coords_list = sampled_data[class_id]
        class_display_name = class_name_mapping.get(class_id, f'Class {class_id}')

        for col_idx in range(max_cols):
            ax = axes[class_idx, col_idx]

            if col_idx < len(coords_list):
                x_coord, y_coord = coords_list[col_idx]
                try:
                    patch = slide.read_region(
                        location=(int(x_coord), int(y_coord)),
                        level=0,
                        size=(patch_size, patch_size)
                    ).convert('RGB')
                    ax.imshow(patch)
                    ax.set_title(f'({x_coord}, {y_coord})', fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
                    print(f"Error reading patch at ({x_coord}, {y_coord}): {e}")
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')
            if col_idx == 0:
                ax.text(-0.1, 0.5, class_display_name, transform=ax.transAxes,
                        va='center', ha='right', fontsize=12, fontweight='bold', rotation=90)

    plt.suptitle('Patch Visualization by Class (Grid Layout)', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


class SlidePatchDataset(Dataset):
    """Reads patches from a WSI on demand. Each DataLoader worker process opens its own
    OpenSlide handle lazily (OpenSlide objects can't be shared/pickled across processes)."""

    def __init__(self, slide_path, coords, transform, patch_size):
        self.slide_path = slide_path
        self.coords = coords
        self.transform = transform
        self.patch_size = patch_size
        self._slide = None

    def _get_slide(self):
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.slide_path)
        return self._slide

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        slide = self._get_slide()
        patch = slide.read_region(
            location=(int(x), int(y)), level=0, size=(self.patch_size, self.patch_size)
        ).convert('RGB')
        patch = self.transform(patch)
        return patch, int(x), int(y)


def process_coords_batch_simple(slide_path, coords, test_transform, model, batch_size=32,
                                 patch_size=512, num_workers=16):
    roi_coords = []
    prediction_results = []
    prediction_roi_results = []

    dataset = SlidePatchDataset(slide_path, coords, test_transform, patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, shuffle=False)

    for patch_tensor, xs, ys in tqdm(loader, desc="Processing batches", leave=False):
        patch_tensor = patch_tensor.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            logits, probs, preds = model(patch_tensor)

        preds_np = preds.cpu().numpy()
        xs_np, ys_np = xs.numpy(), ys.numpy()

        for x, y, pred in zip(xs_np, ys_np, preds_np):
            coord = [int(x), int(y)]
            # ROI classes kept: 0 tumor, 1 tumor-associated stroma, 2 necrosis, 3 inflamed stroma
            # (class 4 = rest, excluded)
            if pred in (0, 1, 2, 3):
                roi_coords.append(coord)
                prediction_roi_results.append(int(pred))

            prediction_results.append(int(pred))

    return roi_coords, prediction_results, prediction_roi_results


def model_init():
    student = Student_convnext_backbone()

    student_projection_head = Student_Projection_Head(
        in_dim=1024, out_dim=1536, use_bn=True, use_mid_layer=False,
        hidden_dim=2048, bottleneck_dim=256, nlayers=3, logger=None,
    )

    student_final = StudentModel_convnext(backbone=student, projection_head=student_projection_head)
    student_final.load_state_dict(torch.load(FEATURE_EXTRACTOR_DIR, map_location='cpu')['student'])
    backbone = student_final.backbone

    model = Patch_Classifier_Softmax_KD(backbone=backbone, num_classes=5).to(DEVICE)
    return model


def main():
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517)
        ),
    ])

    model = model_init()
    ckpt = torch.load(TIGER_MODEL_PATH, map_location='cpu')
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded TIGER classifier from {TIGER_MODEL_PATH}')

    df = pd.read_csv(args.clinical_dir, encoding='cp949')
    listed = sorted([word for word in list(df['tube label'])])

    save_coords_dir = os.path.join(args.output_dir, 'coords')
    save_visu_dir = os.path.join(args.output_dir, 'visu')
    save_pred_dir = os.path.join(args.output_dir, 'predictions')

    for d in (args.output_dir, save_coords_dir, save_visu_dir, save_pred_dir):
        os.makedirs(d, exist_ok=True)

    sample_real = [word[:-4] for word in os.listdir(args.slide_dir)]

    skipped, failed, done = [], [], []

    for idx, sample in enumerate(tqdm(listed, desc='Slides')):

        if sample not in sample_real:
            if sample[0] == '1':
                sample_name = sample.split('_')[0] + '-' + sample.split('_')[1] + '_' + sample.split('_')[2]
            else:
                print(f"SKIPPING {sample} (no matching slide file)")
                skipped.append(sample)
                continue
        else:
            sample_name = sample

        pred_out_path = os.path.join(save_pred_dir, f'{sample_name}.npy')
        if os.path.isfile(pred_out_path):
            continue  # already processed, resume-safe

        slide_path = os.path.join(args.slide_dir, f"{sample_name}.svs")
        coords_path = os.path.join(args.coords_dir, f"{sample_name}.h5")

        if not os.path.isfile(slide_path) or not os.path.isfile(coords_path):
            print(f"SKIPPING {sample_name}: missing slide or coords file")
            skipped.append(sample_name)
            continue

        try:
            with h5py.File(coords_path) as file:
                coords = file['coords'][:]

            roi_coords, prediction_results, prediction_results_roi = process_coords_batch_simple(
                slide_path, coords, test_transform, model, batch_size=args.batch_size,
                patch_size=args.patch_size, num_workers=args.num_workers
            )
            roi_coords_arry = np.array(roi_coords)

            sampled_data = sample_patches_by_class(coords, prediction_results, args.num_vis_samples)
            save_path = os.path.join(save_visu_dir, f'{sample_name}.png')
            slide = openslide.OpenSlide(slide_path)
            visualize_patches_grid(slide, sampled_data, max_cols=args.num_vis_samples,
                                    patch_size=args.patch_size, save_path=save_path)
            slide.close()

            np.save(os.path.join(save_coords_dir, f'{sample_name}.npy'), roi_coords_arry)
            np.save(pred_out_path, np.array(prediction_results_roi))

            done.append(sample_name)
        except Exception as e:
            print(f"FAILED {sample_name}: {e}")
            failed.append(sample_name)

    print(f"\nDone: {len(done)}, Skipped: {len(skipped)}, Failed: {len(failed)}")
    if skipped:
        print("Skipped samples:", skipped)
    if failed:
        print("Failed samples:", failed)


if __name__ == "__main__":
    main()
