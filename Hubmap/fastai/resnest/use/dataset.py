from use.function import img2tensor
from use.config import config
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# default
mean = np.array([0, 0, 0])
std = np.array([1, 1, 1])

class HuBMAPDataset():
    def __init__(self, fold=0, train=True, tfms=None, mean=mean, std=std):
        ids = pd.read_csv(config.LABELS).id.astype(str).values
        kf = KFold(n_splits=config.nfolds,random_state=config.SEED,shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        self.fnames = [fname for fname in os.listdir(config.TRAIN) if fname.split('_')[0] in ids]
        self.train = train
        self.tfms = tfms
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(config.TRAIN,fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(config.MASKS,fname),cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']
        return img2tensor((img/255.0 - mean)/std),img2tensor(mask)