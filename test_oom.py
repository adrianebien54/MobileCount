#!/usr/bin/env python3
"""Test whether MobileCount fits batch_size=8 at 1544x1038 on the GPU."""
import os
import sys
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import torch
from datasets.Tenebrio.loading_data import loading_data
from datasets.Tenebrio.setting import cfg_data
from models.CC import CrowdCounter
from config import cfg

cfg_data.TRAIN_BATCH_SIZE = 8

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

print(f"GPU:        {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Resolution: {cfg_data.DATA_PATH.split('/')[-1]}")
print(f"Batch size: {cfg_data.TRAIN_BATCH_SIZE}")
print()

try:
    net = CrowdCounter([0], 'MobileCount').cuda()
    train_loader, _, _ = loading_data()
    img, gt = next(iter(train_loader))
    print(f"Batch shape: {img.shape}")
    img, gt = img.cuda(), gt.cuda()

    net.train()
    net(img, gt)
    net.loss.backward()

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n✅  No OOM — peak VRAM used: {peak:.2f} GB (batch_size=8, 1544×1038)")

except torch.cuda.OutOfMemoryError as e:
    print(f"\n❌  OOM at batch_size=8, 1544×1038")
    print(f"    {e}")
    sys.exit(1)
