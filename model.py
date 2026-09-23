import torch
import timm 
from torchvision import transforms

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
import os
import torch.distributed as dist

import torchvision
from torch.nn import SyncBatchNorm

class StudentModel_convnext(nn.Module):
    def __init__(self, backbone, projection_head):
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        
    def forward(self, x):
        features = self.backbone(x)
        projected = self.projection_head(features)
        return projected

class Student_Projection_Head(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn, use_mid_layer, hidden_dim, bottleneck_dim, nlayers, logger):
        super().__init__()
        self.use_mid_layer = use_mid_layer
        self.logger = logger
        
        if use_mid_layer:
            self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=True)  # default mlp_bias
            self.apply(self._init_weights)
            self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
        else:
            self.last_layer = weight_norm(nn.Linear(in_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
        
        self.final_batch_norm = SyncBatchNorm(out_dim, affine=False)  # keep SyncBatchNorm in float32 under FSDP


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        # apply the mid layer (mlp) when enabled
        if self.use_mid_layer:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-12)
            x = self.last_layer(x)
        else:
            x = self.last_layer(x)

        #self.logger.info(f"Rank {dist.get_rank()} - before SyncBatchNorm: shape={x.shape}, dtype={x.dtype}")
        x = self.final_batch_norm(x)  # apply SyncBatchNorm
        #self.logger.info(f"Rank {dist.get_rank()} - after SyncBatchNorm: shape={x.shape}, dtype={x.dtype}")
        
        return x

class Student_convnext_backbone(nn.Module):

    def __init__(self, model_path=None, embed_dim=1024):
        super().__init__()
        self.backbone = torchvision.models.convnext_base(pretrained = False)
        if model_path is not None : 
            self.backbone.load_state_dict(torch.load(model_path))
        self.backbone.classifier = torch.nn.Identity()
        self.embed_dim = embed_dim
        
    def forward(self, x):
        
        features = self.backbone(x).squeeze(-1).squeeze(-1)
        return features

def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)

class Patch_Classifier_Softmax_KD(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # output num_classes logits
            nn.Linear(1024, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone.backbone(x)
        logits = self.classifier(features)  # [B, num_classes]
        probs = torch.softmax(logits, dim=-1)  # [B, num_classes]
        preds = torch.argmax(probs, dim=-1)   # [B]
        return logits, probs, preds

import math
from torch import nn

class LoRA_Linear(nn.Module):
    def __init__(self, weight, bias, lora_dim):
        super(LoRA_Linear, self).__init__()

        row, column = weight.shape

        # restore Linear
        if bias is None:
            self.linear = nn.Linear(column, row, bias=False)
            self.linear.load_state_dict({"weight": weight})
        else:
            self.linear = nn.Linear(column, row)
            self.linear.load_state_dict({"weight": weight, "bias": bias})

        # create LoRA weights (with initialization)
        self.lora_right = nn.Parameter(torch.zeros(column, lora_dim))
        #nn.init.kaiming_uniform_(self.lora_right, a=math.sqrt(5))
        nn.init.normal_(self.lora_right, mean=0, std=1)
        self.lora_left = nn.Parameter(torch.zeros(lora_dim, row))

    def forward(self, input):
        x = self.linear(input)
        y = input @ self.lora_right @ self.lora_left
        return x + y

class SoMA_Linear(nn.Module):
    def __init__(self, weight, bias, lora_dim):
        super(SoMA_Linear, self).__init__()

        row, column = weight.shape  # row=out, column=in

        # Compute SVD on the effective weight matrix (weight.T)
        W_eff = weight.t()  # (column, row) = (in, out)
        U, S, Vh = torch.linalg.svd(W_eff, full_matrices=False)  # U (in, min), S (min), Vh (min, out)

        # Assuming min_dim = min(column, row)
        min_dim = min(column, row)

        # Select minor r components (smallest singular values)
        r = lora_dim
        if r > min_dim:
            raise ValueError("lora_dim cannot exceed min(in_features, out_features)")

        U_minor = U[:, -r:]  # (in, r)
        S_minor = S[-r:]  # (r,)
        Vh_minor = Vh[-r:, :]  # (r, out)

        # Compute minor matrix M = U_minor @ diag(S_minor) @ Vh_minor  (in, out)
        D_minor = torch.diag(S_minor)  # (r, r)

        # Residual (major) for effective W_eff
        # But instead of computing full major, subtract minor from original
        minor = U_minor @ D_minor @ Vh_minor  # (in, out)
        major = W_eff - minor  # (in, out)

        # Set the linear weight to major.T (out, in)
        if bias is None:
            self.linear = nn.Linear(column, row, bias=False)
            self.linear.weight.data = major.t()  # (out, in)
        else:
            self.linear = nn.Linear(column, row)
            self.linear.weight.data = major.t()  # (out, in)
            self.linear.bias.data = bias

        # Initialize SoMA (LoRA-like) with minor components
        # lora_right (in, r) = U_minor
        # lora_left (r, out) = D_minor @ Vh_minor
        self.lora_right = nn.Parameter(U_minor)  # (column, r)
        self.lora_left = nn.Parameter(D_minor @ Vh_minor)  # (r, out)

    def forward(self, input):
        x = self.linear(input)
        y = input @ self.lora_right @ self.lora_left
        return x + y

