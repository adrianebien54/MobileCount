from easydict import EasyDict as edict

__C_Tenebrio = edict()
cfg_data = __C_Tenebrio

__C_Tenebrio.DATA_PATH = 'datasets/Tenebrio/772x519'

# ImageNet mean/std — correct for CSRNet's pretrained VGG16 frontend
__C_Tenebrio.MEAN_STD = (
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
)

# Scale density values by 100 so MSE loss is large enough for effective gradient flow.
# Density tensors will sum to count*100; MAE computation divides back by LOG_PARA.
__C_Tenebrio.LOG_PARA = 100.

__C_Tenebrio.RESUME_MODEL = ''

# Full-resolution Tenebrio images are large; batch > 1 will OOM
__C_Tenebrio.TRAIN_BATCH_SIZE = 1
__C_Tenebrio.VAL_BATCH_SIZE = 1
