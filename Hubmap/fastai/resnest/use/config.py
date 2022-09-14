from fastai.vision.all import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from albumentations import *

class config:
    bs = 8
    nfolds = 4
    SEED = 777
    TRAIN = '../512dataset/train'
    MASKS = '../512dataset/masks'
    LABELS = '../train.csv'
    NUM_WORKERS = 2
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    p = 1
    train_transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        # Morphology
        ShiftScaleRotate(shift_limit=(-0.1, 0.1), scale_limit=(-0.2, 0.2), rotate_limit=(-30, 30), interpolation=1, border_mode=0, value=(0, 0, 0), p=0.4),
        GaussNoise(var_limit=(5.0, 50.0), mean=10, p=0.5), # var_limit = 10 -> 5 정도로 바꾸는게 나을듯?
        GaussianBlur(blur_limit=(3, 7), p=0.5),
        # Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
        # Color
        RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5,
                                 brightness_by_max=True, p=0.5),
        HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
                           val_shift_limit=30, p=0.5),
        OneOf([
            OpticalDistortion(p=0.5),
            GridDistortion(p=0.5),
            PiecewiseAffine(p=0.5),
        ], p=0.5),
    ], p=p)
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu" # do not use cuda for out of memory exception in Kaggle
    head_epoch = 4
    head_lr_max = 4e-3

    full_epoch = 32
    full_lr_max = slice(2e-4,2e-3) 

    model = 'Resnest269'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

