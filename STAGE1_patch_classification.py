import argparse
import os
import sys

import numpy as np
import pandas as pd

### PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Sampler
from dataset import DataAugmentationPathologyDINO, PathologyPatchDataset, make_normalize_transform
from model import StudentModel_convnext, Student_Projection_Head, Student_convnext_backbone, Patch_Classifier_Softmax_KD, LoRA_Linear, SoMA_Linear

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import pickle

try:
    import cucim
    CUCIM_AVAILABLE = True
except ImportError:
    CUCIM_AVAILABLE = False

import openslide
from PIL import Image

from typing import Sequence
from torchvision import transforms
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import NMF

from collections import defaultdict
import random

from tensorboardX import SummaryWriter

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

### Training settings
parser = argparse.ArgumentParser(description='STAGE1: weakly-supervised patch-level recurrence-risk classifier.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--dirs', type=str,
                     default=os.path.join(REPO_DIR, 'TIGER_training', 'ROI_sampling_all', 'coords'),
                     help='ROI-filtered coordinate locations (used for periodic test-set evaluation)')
parser.add_argument('--slide_dir', type=str, default='/path/to/your/wsi', help='directory of WSI files (.svs)')
parser.add_argument('--clinical', type=str, default='/path/to/your/clinical_info.csv', help='clinical/outcome CSV (needs at least tube label, Recur, RFS columns)')
parser.add_argument('--train_df_dir', type=str, default=os.path.join(REPO_DIR, '0_folds', 'TNBC_train_df.csv'), help='TNBC train split (zero-fold)')
parser.add_argument('--test_df_dir', type=str, default=os.path.join(REPO_DIR, '0_folds', 'TNBC_test_df.csv'), help='TNBC test split (zero-fold)')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU would be used')
parser.add_argument('--patch_size', type=int, default=512, help='Patch splition size')
parser.add_argument('--train_img_pkl', type=str,
                     default=os.path.join(REPO_DIR, 'patch_info_KBSMC_train_zero_filtered.pkl'),
                     help='patch dataset for training (saved as coordinates)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size during training')
parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size during periodic test-set evaluation (inference-only, does not affect training)')
parser.add_argument('--eval_num_workers', type=int, default=16, help='DataLoader workers for periodic test-set evaluation')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-7, help='weight decay')
parser.add_argument('--writer_dir', type=str, default=os.path.join(REPO_DIR, 'tensorboard_STAGE1'), help='tensorboard logging directory')
parser.add_argument('--weights_dir', type=str, default=os.path.join(REPO_DIR, 'weights_STAGE1'), help='model weights directory per each epochs')
parser.add_argument('--test_interval', type=int, default=1000, help='Doing inference at each test_interval phase')
parser.add_argument('--seed', type=int, default=7, help='Random seed for reproducible experiment (default: 1)')

parser.add_argument('--LoRA', action='store_true', default=False, help='Using LoRA or Not')
parser.add_argument('--lora_dim', type=int, default=16, help='Row rank dimension selection for LoRA and SoMA')
parser.add_argument('--iteration', type=int, default=30000, help='Number of training iterations')

parser.add_argument('--SN', action='store_true', default=False, help='Using Stain Normalization or Not')

parser.add_argument('--early_stop_patience', type=int, default=5,
                     help='Stop after this many consecutive validations without Val/AUC improvement '
                          '(measured in test_interval units). Set to 0 to disable early stopping.')

parser.add_argument('--resume_checkpoint', type=str, default=None,
                     help='Model state_dict path to resume training from (e.g. a previous run\'s checkpoint).')
parser.add_argument('--start_iteration', type=int, default=0,
                     help='global_iteration to resume from (checkpoint numbering / remaining iteration count '
                          'both continue from here). Use together with --resume_checkpoint.')
parser.add_argument('--best_auc_init', type=float, default=-1.0,
                     help='Seed the early-stopping best-AUC tracker with a prior run\'s best Val/AUC, '
                          'so the patience counter is relative to that when resuming.')

args = parser.parse_args()

EPOCH = 1
DEVICE = 'cuda:{}'.format(args.gpu)
IMAGENET_DEFAULT_MEAN = (0.707223, 0.578729, 0.703617)
IMAGENET_DEFAULT_STD = (0.211883, 0.230117, 0.177517)
FEATURE_EXTRACTOR_DIR = os.path.join(REPO_DIR, 'G2B_BRCA.pth')

target_img = Image.open(os.path.join(REPO_DIR, 'source.png')) if args.SN else None


def rgb2od(img):
    """RGB -> Optical Density"""
    img = img.astype(np.float32)
    od = -np.log((img + 1.0) / 240.0)
    return od


def od2rgb(od):
    """Optical Density -> RGB"""
    rgb = np.exp(-od) * 240.0 - 1.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


class VahadaneNormalizerFast:
    def __init__(self, lambda_val=0.1, target_mask=None,
                 od_threshold=0.15, nmf_max_iter=200, nmf_tol=1e-6,
                 nmf_samples=20000, random_state=0):
        self.lambda_val = lambda_val
        self.target_mask = target_mask
        self.od_threshold = od_threshold
        self.nmf_max_iter = nmf_max_iter
        self.nmf_tol = nmf_tol
        self.nmf_samples = nmf_samples
        self.random_state = random_state
        self.source_stain_matrix = None
        self.target_stain_matrix = None

    def _tissue_pixels_from_od(self, od_flat, mask_flat=None):
        if mask_flat is not None:
            idx = (mask_flat == 255)
            od_sel = od_flat[idx]
        else:
            od_sel = od_flat

        tissue_idx = np.all(od_sel > self.od_threshold, axis=1)
        od_tissue = od_sel[tissue_idx]

        if od_tissue.shape[0] < 100:
            tissue_idx2 = np.any(od_sel > 0.05, axis=1)
            od_tissue = od_sel[tissue_idx2]

        return od_tissue

    def _sample_rows(self, X, max_n):
        n = X.shape[0]
        if n <= max_n:
            return X
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=max_n, replace=False)
        return X[idx]

    def _extract_stain_matrix_nmf(self, img, mask=None):
        od = rgb2od(img)
        od_flat = od.reshape(-1, 3)

        mask_flat = None
        if mask is not None:
            mask_flat = mask.reshape(-1)

        od_tissue = self._tissue_pixels_from_od(od_flat, mask_flat=mask_flat)
        if od_tissue.shape[0] < 50:
            raise ValueError("Too few tissue pixels to estimate a stain matrix.")

        od_tissue = self._sample_rows(od_tissue, self.nmf_samples)

        nmf = NMF(
            n_components=2,
            init="nndsvda",
            tol=self.nmf_tol,
            max_iter=self.nmf_max_iter,
            random_state=self.random_state
        )
        W = nmf.fit_transform(od_tissue)
        H = nmf.components_

        if H.shape != (2, 3):
            if H.shape == (3, 2):
                H = H.T
            else:
                raise ValueError(f"Unexpected stain matrix shape: {H.shape}")

        return H.astype(np.float32)

    def fit(self, target_img):
        self.target_stain_matrix = self._extract_stain_matrix_nmf(target_img, mask=self.target_mask)
        return self

    def transform(self, source_img):
        if self.target_stain_matrix is None:
            raise ValueError("Call fit(target_img) first.")

        self.source_stain_matrix = self._extract_stain_matrix_nmf(source_img, mask=None)

        source_od = rgb2od(source_img)
        h, w, _ = source_od.shape
        source_od_flat = source_od.reshape(-1, 3).astype(np.float32)

        tissue_idx = np.any(source_od_flat > self.od_threshold, axis=1)
        B = source_od_flat[tissue_idx]

        A = self.source_stain_matrix.T
        A_pinv = np.linalg.pinv(A).astype(np.float32)

        C = (A_pinv @ B.T).T
        C = np.clip(C, 0, None)

        source_conc = np.zeros((source_od_flat.shape[0], 2), dtype=np.float32)
        source_conc[tissue_idx] = C

        normalized_od_flat = source_conc @ self.target_stain_matrix
        normalized_od = normalized_od_flat.reshape(h, w, 3)

        normalized_img = od2rgb(normalized_od)
        return normalized_img


def normalize_image(source_img, target_img, target_mask):
    normalizer = VahadaneNormalizerFast(target_mask=target_mask)
    normalizer.fit(target_img)
    return normalizer.transform(source_img)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def Patch_Classifier():
    student = Student_convnext_backbone()

    student_projection_head = Student_Projection_Head(
        in_dim=1024, out_dim=1536, use_bn=True, use_mid_layer=False,
        hidden_dim=2048, bottleneck_dim=256, nlayers=3, logger=None,
    )

    backbone = StudentModel_convnext(backbone=student, projection_head=student_projection_head)
    backbone.load_state_dict(torch.load(FEATURE_EXTRACTOR_DIR, map_location='cpu')['student'])
    model = Patch_Classifier_Softmax_KD(backbone=backbone)

    return model


def LoRA_application(args, model):
    target_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ('features.3' in name or 'features.5' in name or 'features.7' in name):
            target_names.append(name)

    dim = args.lora_dim
    for name in target_names:
        name_struct = name.split(".")
        module_list = [model]
        for struct in name_struct:
            module_list.append(getattr(module_list[-1], struct))
        lora = LoRA_Linear(
            weight=module_list[-1].weight,
            bias=module_list[-1].bias,
            lora_dim=dim,
        ).to(DEVICE)
        module_list[-2].__setattr__(name_struct[-1], lora)

    for name, param in model.named_parameters():
        if 'lora' not in name:
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True

    return model


class BalancedBatchSampler(Sampler):
    """
    - Label (recurrence/non-recurrence): balanced within each batch (50:50)
    - Category: natural proportion preserved within each label
    - Patient: sampled round-robin across patients
    """
    def __init__(self, patch_info, batch_size, num_iterations=10000):
        self.batch_size = batch_size
        self.num_iterations = num_iterations

        self.groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for idx, info in enumerate(patch_info):
            label = info['label']
            cat = info['category']
            patient = info['slide_name']
            self.groups[label][cat][patient].append(idx)

        self.labels = sorted(self.groups.keys())
        self.per_label = batch_size // len(self.labels)

        self.label_cat_allocation = {}
        for label in self.labels:
            cats = sorted(self.groups[label].keys())
            cat_sizes = {
                cat: sum(len(v) for v in self.groups[label][cat].values())
                for cat in cats
            }
            total = sum(cat_sizes.values())

            raw_alloc = {cat: (cat_sizes[cat] / total) * self.per_label for cat in cats}

            int_alloc = {cat: int(raw_alloc[cat]) for cat in cats}
            remainder = self.per_label - sum(int_alloc.values())
            fractional = {cat: raw_alloc[cat] - int_alloc[cat] for cat in cats}
            for cat in sorted(fractional, key=fractional.get, reverse=True):
                if remainder <= 0:
                    break
                int_alloc[cat] += 1
                remainder -= 1

            self.label_cat_allocation[label] = {
                cat: n for cat, n in int_alloc.items() if n > 0
            }

        print(f"[BalancedBatchSampler] labels={self.labels}, per_label={self.per_label}")
        for label in self.labels:
            cat_sizes = {
                cat: sum(len(v) for v in self.groups[label][cat].values())
                for cat in self.groups[label]
            }
            print(f"  label={label}: total={sum(cat_sizes.values())}")
            print(f"    category sizes: {cat_sizes}")
            print(f"    batch allocation: {self.label_cat_allocation[label]}")
        print(f"  num_iterations={self.num_iterations}")

    def _make_queue(self, label, cat):
        patient_dict = self.groups[label][cat]
        patients = list(patient_dict.keys())
        random.shuffle(patients)

        patient_pools = {}
        for p in patients:
            pool = list(patient_dict[p])
            random.shuffle(pool)
            patient_pools[p] = pool

        queue = []
        patient_iters = {p: iter(pool) for p, pool in patient_pools.items()}
        active = list(patients)

        while active:
            next_active = []
            for p in active:
                try:
                    queue.append(next(patient_iters[p]))
                    next_active.append(p)
                except StopIteration:
                    pass
            active = next_active

        return queue

    def __iter__(self):
        queues = {}
        pointers = {}
        for label in self.labels:
            for cat in self.label_cat_allocation[label]:
                key = (label, cat)
                queues[key] = self._make_queue(label, cat)
                pointers[key] = 0

        for _ in range(self.num_iterations):
            batch = []
            for label in self.labels:
                for cat, n in self.label_cat_allocation[label].items():
                    key = (label, cat)
                    for _ in range(n):
                        if pointers[key] >= len(queues[key]):
                            queues[key] = self._make_queue(label, cat)
                            pointers[key] = 0
                        batch.append(queues[key][pointers[key]])
                        pointers[key] += 1

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_iterations


class EvalPatchDataset(Dataset):
    """Parallelized replacement for the original evaluation()'s per-patch sequential read_region
    loop. Each DataLoader worker opens its own OpenSlide handle lazily."""

    def __init__(self, slide_path, coords, transform, use_sn=False):
        self.slide_path = slide_path
        self.coords = coords
        self.transform = transform
        self.use_sn = use_sn
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

        if self.use_sn:
            patch = normalize_image(np.array(patch), np.array(target_img), target_mask=None)
            patch = Image.fromarray(patch)

        return self.transform(patch)


def evaluation(model, dirs, df, eval_transform, criterion, topk=20):
    model.eval()

    labels = []
    high_prob_scores = []

    slide_prob_scores_agg = {0: [], 1: []}
    slide_prob_scores_per_patient = {}

    losses = []

    sample_real = [word[:-4] for word in os.listdir(dirs)]

    for sample in tqdm(list(df['tube label']), desc='Eval slides'):
        try:
            slide_prob_scores_per_patient[sample] = []

            if sample not in sample_real:
                if sample[0] == '1':
                    sample_name = sample.split('_')[0] + '-' + sample.split('_')[1] + '_' + sample.split('_')[2]
                else:
                    print("SKIPPING {}".format(sample))
                    continue
            else:
                sample_name = sample

            coords_list = np.load(os.path.join(dirs, sample_name + '.npy'))
            slide_path = os.path.join(args.slide_dir, f"{sample_name}.svs")

            label = df[df['tube label'] == sample]['Recur'].item()
            labels.append(label)

            ds = EvalPatchDataset(slide_path, coords_list, eval_transform, use_sn=args.SN)
            loader = DataLoader(ds, batch_size=args.eval_batch_size, num_workers=args.eval_num_workers,
                                 pin_memory=True, shuffle=False)

            slide_risk_scores = []
            for batch_tensor in loader:
                batch_tensor = batch_tensor.to(DEVICE, non_blocking=True)
                with torch.no_grad():
                    logits, probs, preds = model(batch_tensor)
                y_batch = torch.LongTensor([label] * batch_tensor.shape[0]).to(DEVICE)
                loss = criterion(logits, y_batch)
                prob_pos = probs[:, 1].detach().cpu().numpy()
                slide_risk_scores.extend(list(prob_pos))
                slide_prob_scores_agg[label].extend(list(prob_pos))
                slide_prob_scores_per_patient[sample].extend(list(prob_pos))
                losses.append(loss.item())

            high_prob_scores.append(np.mean(sorted(slide_risk_scores, reverse=True)[:topk]))
        except Exception as e:
            print("SKIPPING {}: {}".format(sample, e))

    return slide_prob_scores_agg, high_prob_scores, labels, slide_prob_scores_per_patient, np.mean(losses)


def train(args, train_loader, model, criterion, optimizer, writer, df, eval_transform):
    global_iteration = args.start_iteration
    best_auc = args.best_auc_init
    best_iteration = args.start_iteration if args.best_auc_init > -1.0 else -1
    no_improve_count = 0

    for epoch in range(EPOCH):
        model.train()
        train_loss = []
        stop_early = False

        for batch_idx, (patch, label, name) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCH}')):

            patch = patch.to(DEVICE)
            y = label.to(DEVICE).long()

            logits, probs, preds = model(patch)
            loss = criterion(logits, y)

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_iteration += 1

            if global_iteration % args.test_interval == 0:

                print(f"\n=== Validation at iteration {global_iteration} ===")
                scores_agg, prob_scores, labels, slide_prob_scores_per_patient, loss_eval = evaluation(
                    model, args.dirs, df, eval_transform, criterion
                )

                auc = roc_auc_score(labels, prob_scores)
                emd = wasserstein_distance(scores_agg[0], scores_agg[1])
                ks_stat, ks_p = ks_2samp(scores_agg[0], scores_agg[1])

                train_loss_all = np.mean(train_loss)

                writer.add_scalar('Train/Loss', train_loss_all, global_iteration)
                writer.add_scalar('Val/Loss', loss_eval, global_iteration)
                writer.add_scalar('Val/AUC', auc, global_iteration)
                writer.add_scalar('Val/KS Stat', ks_stat, global_iteration)
                writer.add_scalar('Val/Wasserstein D', emd, global_iteration)

                print(f"  Val/AUC={auc:.4f} Val/KS={ks_stat:.4f} Val/EMD={emd:.4f} Val/Loss={loss_eval:.4f}")

                torch.save(model.state_dict(), os.path.join(args.weights_dir, f"checkpoint_{global_iteration}.pth"))

                if auc > best_auc:
                    best_auc = auc
                    best_iteration = global_iteration
                    no_improve_count = 0
                    torch.save(model.state_dict(), os.path.join(args.weights_dir, "best_model.pth"))
                    print(f"  -> new best Val/AUC={best_auc:.4f} at iteration {best_iteration}, saved best_model.pth")
                else:
                    no_improve_count += 1
                    print(f"  no improvement for {no_improve_count}/{args.early_stop_patience} validations "
                          f"(best Val/AUC={best_auc:.4f} at iteration {best_iteration})")

                model.train()
                print("=" * 50)
                train_loss = []

                if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                    print(f"Early stopping at iteration {global_iteration} "
                          f"(no Val/AUC improvement for {args.early_stop_patience} validations). "
                          f"Best Val/AUC={best_auc:.4f} at iteration {best_iteration}.")
                    stop_early = True
                    break

        if stop_early:
            break

    print(f"\nTraining finished. global_iteration={global_iteration}, "
          f"best Val/AUC={best_auc:.4f} at iteration {best_iteration}")


def main(args):

    set_seed(args.seed)

    data_transform = DataAugmentationPathologyDINO(global_crops_size=args.patch_size)
    eval_transform = transforms.Compose([
        transforms.Resize((args.patch_size, args.patch_size)),
        transforms.ToTensor(),
        make_normalize_transform(),
    ])

    with open(args.train_img_pkl, "rb") as f:
        patch_info = pickle.load(f)

    test_df = pd.read_csv(args.test_df_dir, encoding='cp949')

    dataset = PathologyPatchDataset(
        root_path=args.slide_dir,
        patch_info=patch_info, img_size=args.patch_size,
        transform=data_transform, cache_size=100,
        use_sn=args.SN
    )

    remaining_iterations = max(0, args.iteration - args.start_iteration)
    sampler = BalancedBatchSampler(
        patch_info=patch_info,
        batch_size=args.batch_size,
        num_iterations=remaining_iterations
    )
    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4
    )

    model = Patch_Classifier().to(DEVICE)

    if args.resume_checkpoint:
        model.load_state_dict(torch.load(args.resume_checkpoint, map_location='cpu'))
        model.to(DEVICE)
        print(f"Resumed model weights from {args.resume_checkpoint} (start_iteration={args.start_iteration})")

    if args.LoRA:
        model = LoRA_application(args, model)
        print("LoRA applicated")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)

    os.makedirs(args.writer_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    writer = SummaryWriter(args.writer_dir, flush_secs=15)

    train(args, train_loader, model, criterion, optimizer, writer, test_df, eval_transform)


if __name__ == "__main__":
    main(args)
