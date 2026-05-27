import os
import argparse
import time
import numpy as np
import torch

from config import cfg

#------------prepare environment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus) == 1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True

#------------parse CLI overrides (must happen before Trainer import)------------
parser = argparse.ArgumentParser(description='MobileCount Training')
parser.add_argument('--lr',           type=float, default=None,
                    help='Override learning rate from config.py')
parser.add_argument('--batch-size',   type=int,   default=None, dest='batch_size',
                    help='Override training batch size from setting.py')
parser.add_argument('--weight-decay', type=float, default=None, dest='weight_decay',
                    help='Override weight decay (currently hardcoded in trainer.py)')
parser.add_argument('--aug-set',      type=int,   default=None, dest='aug_set',
                    help='Augmentation set: 0=none, 1=full suite, 2=flips only')
parser.add_argument('--data-path',    type=str,   default=None, dest='data_path',
                    help='Override DATA_PATH in setting.py (e.g. datasets/Tenebrio/386x260)')
parser.add_argument('--resume',       action='store_true',
                    help='Resume training from a checkpoint')
parser.add_argument('--resume-path',  type=str,   default=None, dest='resume_path',
                    help='Path to latest_state.pth to resume from')
args = parser.parse_args()

#------------select dataset------------
data_mode = cfg.DATASET
if data_mode == 'SHHA':
    from datasets.SHHA.loading_data import loading_data
    from datasets.SHHA.setting import cfg_data
elif data_mode == 'SHHB':
    from datasets.SHHB.loading_data import loading_data
    from datasets.SHHB.setting import cfg_data
elif data_mode == 'QNRF':
    from datasets.QNRF.loading_data import loading_data
    from datasets.QNRF.setting import cfg_data
elif data_mode == 'UCF50':
    from datasets.UCF50.loading_data import loading_data
    from datasets.UCF50.setting import cfg_data
elif data_mode == 'WE':
    from datasets.WE.loading_data import loading_data
    from datasets.WE.setting import cfg_data
elif data_mode == 'GCC':
    from datasets.GCC.loading_data import loading_data
    from datasets.GCC.setting import cfg_data
elif data_mode == 'Tenebrio':
    from datasets.Tenebrio.loading_data import loading_data
    from datasets.Tenebrio.setting import cfg_data

# Apply CLI overrides to mutable singletons before Trainer is constructed
if args.lr is not None:
    cfg.LR = args.lr
if args.batch_size is not None:
    cfg_data.TRAIN_BATCH_SIZE = args.batch_size
if args.aug_set is not None:
    cfg.AUG_SET = args.aug_set
if args.data_path is not None:
    cfg_data.DATA_PATH = args.data_path
if args.resume:
    cfg.RESUME = True
if args.resume_path is not None:
    cfg.RESUME_PATH = args.resume_path

# Regenerate EXP_NAME for fresh runs only (resume loads exp_name from checkpoint)
if not args.resume and any(v is not None for v in [args.lr, args.batch_size, args.weight_decay, args.aug_set, args.data_path]):
    now = time.strftime("%m-%d_%H-%M", time.localtime())
    res = cfg_data.DATA_PATH.rstrip('/').split('/')[-1]
    aug_suffix = f'_aug{cfg.AUG_SET}' if cfg.AUG_SET > 0 else ''
    cfg.EXP_NAME = f"{now}_{cfg.DATASET}_{cfg.NET}_{cfg.LR}_{res}{aug_suffix}"

#------------Prepare Trainer------------
from trainer import Trainer

#------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer(loading_data, cfg_data, pwd)
cc_trainer.forward()
