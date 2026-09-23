from torchvision import transforms
from typing import Sequence
import torch
import openslide
from torch.utils.data import Dataset
import os

import collections
import numpy as np
from PIL import Image
import h5py
import json
import sqlite3
from pathlib import Path
import threading

try:
    import cucim
    CUCIM_AVAILABLE = True
except ImportError:
    CUCIM_AVAILABLE = False

IMAGENET_DEFAULT_MEAN = (0.707223, 0.578729, 0.703617)
IMAGENET_DEFAULT_STD = (0.211883, 0.230117, 0.177517)

import numpy as np
from sklearn.decomposition import NMF
from skimage import color

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
        """
        Faster Vahadane-like normalizer:
          - Target stain basis from (masked) tissue OD pixels via NMF
          - Source stain basis via NMF (sampled)
          - Concentrations via vectorized least squares (pinv) + nonneg clip
        """
        self.lambda_val = lambda_val  # not used directly in this implementation; kept for interface compatibility
        self.target_mask = target_mask

        self.od_threshold = od_threshold
        self.nmf_max_iter = nmf_max_iter
        self.nmf_tol = nmf_tol
        self.nmf_samples = nmf_samples
        self.random_state = random_state

        self.source_stain_matrix = None  # (2,3)
        self.target_stain_matrix = None  # (2,3)

    def _tissue_pixels_from_od(self, od_flat, mask_flat=None):
        """
        od_flat: (N,3)
        mask_flat: (N,) optional. If provided, keep mask_flat==255 first.
        """
        if mask_flat is not None:
            idx = (mask_flat == 255)
            od_sel = od_flat[idx]
        else:
            od_sel = od_flat

        # OD-based tissue filter
        tissue_idx = np.all(od_sel > self.od_threshold, axis=1)
        od_tissue = od_sel[tissue_idx]

        # relax the criterion if too few tissue pixels remain
        if od_tissue.shape[0] < 100:
            # relaxed criterion: any channel with a small amount of OD
            tissue_idx2 = np.any(od_sel > 0.05, axis=1)
            od_tissue = od_sel[tissue_idx2]

        return od_tissue

    def _sample_rows(self, X, max_n):
        """Randomly sample at most max_n rows from X"""
        n = X.shape[0]
        if n <= max_n:
            return X
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=max_n, replace=False)
        return X[idx]

    def _extract_stain_matrix_nmf(self, img, mask=None):
        """
        img: RGB (H,W,3)
        mask: optional (H,W) with 255 for tissue
        return: H (2,3)
        """
        od = rgb2od(img)
        od_flat = od.reshape(-1, 3)

        mask_flat = None
        if mask is not None:
            mask_flat = mask.reshape(-1)

        od_tissue = self._tissue_pixels_from_od(od_flat, mask_flat=mask_flat)
        if od_tissue.shape[0] < 50:
            raise ValueError("Too few tissue pixels to estimate a stain matrix.")

        # subsample before NMF for speed/stability
        od_tissue = self._sample_rows(od_tissue, self.nmf_samples)

        nmf = NMF(
            n_components=2,
            init="nndsvda",
            tol=self.nmf_tol,
            max_iter=self.nmf_max_iter,
            random_state=self.random_state
        )
        W = nmf.fit_transform(od_tissue)
        H = nmf.components_  # (2,3)

        # fix up the shape just in case
        if H.shape != (2, 3):
            if H.shape == (3, 2):
                H = H.T
            else:
                raise ValueError(f"Unexpected stain matrix shape: {H.shape}")

        return H.astype(np.float32)

    def fit(self, target_img):
        """
        If target_mask is provided, estimate the stain basis only from the mask==255 region.
        """
        self.target_stain_matrix = self._extract_stain_matrix_nmf(
            target_img,
            mask=self.target_mask
        )
        return self

    def transform(self, source_img):
        if self.target_stain_matrix is None:
            raise ValueError("Call fit(target_img) first.")

        # 1) estimate the source stain basis (NMF, subsampled)
        self.source_stain_matrix = self._extract_stain_matrix_nmf(source_img, mask=None)  # (2,3)

        # 2) convert to OD
        source_od = rgb2od(source_img)
        h, w, _ = source_od.shape
        source_od_flat = source_od.reshape(-1, 3).astype(np.float32)

        # 3) compute concentrations for tissue pixels only (skip background)
        tissue_idx = np.any(source_od_flat > self.od_threshold, axis=1)
        B = source_od_flat[tissue_idx]  # (Ntissue,3)

        # 4) replace the per-pixel NNLS loop with a vectorized solve:
        #    A c ≈ b, A=(3,2) => c=(2,)
        #    c = pinv(A) b (vectorized), then clip negatives to 0
        A = self.source_stain_matrix.T  # (3,2)
        A_pinv = np.linalg.pinv(A).astype(np.float32)  # (2,3), computed once

        C = (A_pinv @ B.T).T           # (Ntissue,2)
        C = np.clip(C, 0, None)        # non-negativity

        # assemble the full per-pixel concentration matrix
        source_conc = np.zeros((source_od_flat.shape[0], 2), dtype=np.float32)
        source_conc[tissue_idx] = C

        # 5) reconstruct using the target stain basis
        normalized_od_flat = source_conc @ self.target_stain_matrix  # (N,3)
        normalized_od = normalized_od_flat.reshape(h, w, 3)

        # 6) convert back to RGB
        normalized_img = od2rgb(normalized_od)
        return normalized_img

def normalize_image(source_img, target_img, target_mask):
    normalizer = VahadaneNormalizerFast(target_mask=target_mask)
    normalizer.fit(target_img)
    return normalizer.transform(source_img)


class LRUCache:
    """Least Recently Used Cache optimization (LRU Cache) for limitation of memory usage"""
    def __init__(self, capacity):
        self.cache = collections.OrderedDict()
        self.capacity = capacity
        
    def get(self, key):
        if key not in self.cache:
            return None
        # move used items to the back (recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def put(self, key, value):
        # if the key already exists, update it and move it to the back of the list
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        # remove oldest items when over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

class DataAugmentationPathologyDINO(object):
    def __init__(
        self,
        global_crops_size=512
    ):
        global_crops_scale =  [0.99, 1.0]
        self.global_crops_scale = global_crops_scale
        self.global_crops_size = global_crops_size

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),  
            ]
        )

        # Color augmentation for better in pathology image
        # weakr augmentation for jittering
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02)],
                    p=0.5,  # diminish probability
                ),
            ]
        )

        global_transfo_extra = GaussianBlur(p=0.1)
        
        # normalization 
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),  
            ]
        )

        self.global_transfo = transforms.Compose([color_jittering, global_transfo_extra, self.normalize])
        
    def __call__(self, image):
        output = {}

        # global crops:
        im_base = self.geometric_augmentation_global(image)
        global_crop = self.global_transfo(im_base)
        return global_crop

class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)
    
class PathologyPatchDataset(Dataset):
    def __init__(self, root_path, patch_info, img_size, transform=None, cache_size=10, use_sn=False):
        """
        Standard pathology patch dataset with in-memory slide grouping
        
        Parameters:
            root_path (str): WSI dataset directory
            patch_info (list): Patch information files containing slide names and coordinates
            img_size (int): Size of the cropped patches (default: 224)
            transform: DINO augmentation transforms for data augmentation
            cache_size (int): Maximum number of slides to keep in LRU cache for memory efficiency
        """
        self.root_path = root_path
        self.img_size = img_size
        self.transform = transform
        
        self.slides_info = {}
        self.patch_indices = []
        
        # NEW: Collect all labels for easy access (e.g., for sampler weights)
        self.labels = []  # List of labels corresponding to each patch index
        
        self.use_sn = use_sn
        
        # Group each patch information by slide
        for idx, info in enumerate(patch_info):
            slide_name = info['slide_name']
            if slide_name not in self.slides_info:
                self.slides_info[slide_name] = []
            
            # Save patch information (save memory by saving only coordinates)
            self.slides_info[slide_name].append({
                'x': info['x'], 
                'y': info['y'],
                'original_idx': idx, 
                'label': info['label'],
                'img_name': info['img_name'],
                #'category' : info['category']
            })
            
            # Index mapping of the entire dataset
            self.patch_indices.append((slide_name, len(self.slides_info[slide_name]) - 1))
            
            # NEW: Append the label for this patch
            self.labels.append(info['label'])
        
        # Create slide LRU cache (to limit memory usage)
        self.slide_cache = LRUCache(cache_size)
        
    def _get_slide(self, slide_name):

        slide = self.slide_cache.get(slide_name)
        if slide is not None:
            return slide
            
        slide_path = os.path.join(self.root_path, slide_name+'.svs')
        try:
            if CUCIM_AVAILABLE:
                slide = cucim.CuImage(slide_path)
            else:
                slide = openslide.OpenSlide(slide_path)
        except Exception as e:
            try:
                slide = openslide.OpenSlide(slide_path)
            except Exception as inner_e:
                raise Exception(f"Failed to open slide {slide_name}: {e}, {inner_e}")

        self.slide_cache.put(slide_name, slide)
        return slide

    def __getitem__(self, idx):
        slide_name, patch_idx = self.patch_indices[idx]
        patch_info = self.slides_info[slide_name][patch_idx]
        
        slide = self._get_slide(slide_name)
    
        #x, y, label, img_name, category = patch_info['x'], patch_info['y'], patch_info['label'], patch_info['img_name'], patch_info['category']
        x, y, label, img_name = patch_info['x'], patch_info['y'], patch_info['label'], patch_info['img_name']
        
        try:
            patch_origin = Image.fromarray(np.array(slide.read_region(
                location=(x, y), 
                level=0, 
                size=(self.img_size, self.img_size),
                num_workers=4 if CUCIM_AVAILABLE else None
            ))).convert("RGB")
        except:
            patch_origin = Image.fromarray(np.array(slide.read_region(
                location=(x, y), 
                level=0, 
                size=(self.img_size, self.img_size)
            ))).convert("RGB")
        
        if self.use_sn : 
            #print(np.array(patch_origin).shape)
            #print(np.array(target_img).shape)
            patch_origin = normalize_image(np.array(patch_origin), np.array(target_img),target_mask=None)
            patch_origin = Image.fromarray(patch_origin)
        
        if self.transform:
            patch = self.transform(patch_origin)
        else:
            patch = patch_origin
            
        #return patch, label, img_name, category
        return patch, label, img_name

    def __len__(self):
        return len(self.patch_indices)
