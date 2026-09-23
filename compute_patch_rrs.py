"""
Computes a recurrence-risk score (RRS) for every ROI patch of every patient (train + test) using
a trained STAGE1 checkpoint, and saves the result as a patient -> [RRS...] dict pickle. This is the
step that builds the input data for STAGE2 (histogram + Lasso aggregation).
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import openslide

from model import StudentModel_convnext, Student_Projection_Head, Student_convnext_backbone, Patch_Classifier_Softmax_KD

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Compute per-patch RRS for all TNBC patients from a STAGE1 checkpoint.')
parser.add_argument('--checkpoint', type=str,
                     default=os.path.join(REPO_DIR, 'weights_STAGE1', 'checkpoint_9000.pth'))
parser.add_argument('--feature_extractor', type=str, default=os.path.join(REPO_DIR, 'G2B_BRCA.pth'))
parser.add_argument('--train_df_dir', type=str, default=os.path.join(REPO_DIR, '0_folds', 'TNBC_train_df.csv'))
parser.add_argument('--test_df_dir', type=str, default=os.path.join(REPO_DIR, '0_folds', 'TNBC_test_df.csv'))
parser.add_argument('--coords_dir', type=str,
                     default=os.path.join(REPO_DIR, 'TIGER_training', 'ROI_sampling_all', 'coords'))
parser.add_argument('--slide_dir', type=str, default='/path/to/your/wsi')
parser.add_argument('--out_pkl', type=str,
                     default=os.path.join(REPO_DIR, 'slide_prob_scores_per_patient_iter9000.pkl'))
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--patch_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=16)
args = parser.parse_args()

DEVICE = f'cuda:{args.gpu}'
IMAGENET_DEFAULT_MEAN = (0.707223, 0.578729, 0.703617)
IMAGENET_DEFAULT_STD = (0.211883, 0.230117, 0.177517)


def Patch_Classifier():
    student = Student_convnext_backbone()
    student_projection_head = Student_Projection_Head(
        in_dim=1024, out_dim=1536, use_bn=True, use_mid_layer=False,
        hidden_dim=2048, bottleneck_dim=256, nlayers=3, logger=None,
    )
    backbone = StudentModel_convnext(backbone=student, projection_head=student_projection_head)
    backbone.load_state_dict(torch.load(args.feature_extractor, map_location='cpu')['student'])
    model = Patch_Classifier_Softmax_KD(backbone=backbone)
    return model


class EvalPatchDataset(Dataset):
    def __init__(self, slide_path, coords, transform):
        self.slide_path = slide_path
        self.coords = coords
        self.transform = transform
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
        patch = slide.read_region(location=(int(x), int(y)), level=0, size=(512, 512)).convert('RGB')
        return self.transform(patch)


def resolve_slide_name(tube_label, slide_name_reals):
    if tube_label in slide_name_reals:
        return tube_label
    if tube_label[0] == '1':
        parts = tube_label.split('_')
        return parts[0] + '-' + parts[1] + '_' + parts[2]
    return None


def main():
    eval_transform = transforms.Compose([
        transforms.Resize((args.patch_size, args.patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    model = Patch_Classifier().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    train_df = pd.read_csv(args.train_df_dir, encoding='cp949')
    test_df = pd.read_csv(args.test_df_dir, encoding='cp949')
    all_patients = list(train_df['tube label']) + list(test_df['tube label'])
    print(f"train: {len(train_df)}, test: {len(test_df)}, total: {len(all_patients)}")

    sample_real = [w[:-4] for w in os.listdir(args.slide_dir)]

    slide_prob_scores_per_patient = {}
    skipped = []

    for sample in tqdm(all_patients, desc='Patients'):
        sample_name = resolve_slide_name(sample, sample_real)
        if sample_name is None:
            print(f"SKIPPING {sample} (no matching slide file)")
            skipped.append(sample)
            continue

        coords_path = os.path.join(args.coords_dir, sample_name + '.npy')
        slide_path = os.path.join(args.slide_dir, f"{sample_name}.svs")
        if not os.path.isfile(coords_path) or not os.path.isfile(slide_path):
            print(f"SKIPPING {sample_name} (missing coords/slide)")
            skipped.append(sample_name)
            continue

        coords = np.load(coords_path)
        ds = EvalPatchDataset(slide_path, coords, eval_transform)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=True, shuffle=False)

        scores = []
        for batch in loader:
            batch = batch.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                logits, probs, preds = model(batch)
            scores.extend(probs[:, 1].detach().cpu().numpy().tolist())

        slide_prob_scores_per_patient[sample] = scores

    with open(args.out_pkl, 'wb') as f:
        pickle.dump(slide_prob_scores_per_patient, f)

    print(f"\nSaved RRS scores for {len(slide_prob_scores_per_patient)} patients to {args.out_pkl}")
    print(f"Skipped: {len(skipped)} -> {skipped}")


if __name__ == "__main__":
    main()
