import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse
from use.function import *

from fastai.vision.all import *
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import cv2
import gc
import random
from albumentations import *
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from use.config import *
from use.dataset import *

from fastai.callback.tracker import TrackerCallback
from fastcore.basics import store_attr

from use.models import *
import shutil

"""# Train"""
seed_everything(config.SEED)

gc.collect()

mean = np.array([0.69646434, 0.67500444, 0.69009685])
std = np.array([0.33764726, 0.34157381, 0.34228867])

dice = Dice_th_pred(np.arange(0.2,0.7,0.01))

if not os.path.exists("models"):
        os.mkdir("models")

fold_cnt = int(config.nfolds)

for fold in range(0, fold_cnt):
    if not os.path.exists(f"models/fold_{fold}"):
        os.mkdir(f"models/fold_{fold}")
    ds_t = HuBMAPDataset(fold=fold, train=True, tfms=config.train_transform, mean=mean, std=std)
    ds_v = HuBMAPDataset(fold=fold, train=False)
    data = ImageDataLoaders.from_dsets(ds_t,ds_v,bs=config.bs,
                num_workers=config.NUM_WORKERS,pin_memory=True)
    model = UneXt50().cuda()
    # if want to train continually, use this 
    # model.load_state_dict(torch.load('weight path'))
    if config.device != "cpu":
        data.to(config.device)
        model.to(config.device)
    

    learn = Learner(data, model, loss_func=symmetric_lovasz,
                metrics=[Dice_soft(),Dice_th()], 
                splitter=split_layers).to_fp16()     
    
    
    learn.freeze_to(-1) #doesn't work
    for param in learn.opt.param_groups[0]['params']:
        param.requires_grad = False
    learn.fit_one_cycle(config.head_epoch, lr_max=config.head_lr_max)

    #continue training full model
    learn.unfreeze()
    learn.fit_one_cycle(config.full_epoch, lr_max=config.full_lr_max,
                        cbs=MySaveModelCallback(monitor='dice_soft', comp=np.greater,
                                              fname=f'fold_{fold}/{config.model}_fold_{fold}',
                                              every_epoch=False))
    
    
    save_path = './output/'
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    torch.save(learn.model.state_dict(), save_path + f'model_{fold}.pth')
    
    save_dir = save_path
    name = sorted(os.listdir(f'./models/fold_{fold}'))[-1]
    f_name = f'./models/fold_{fold}/' + name

    import shutil
    shutil.copy(f_name, save_dir+name)





# config에서 경로 설정
