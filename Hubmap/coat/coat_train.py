# !pip install rasterio
# !pip install -U albumentations
# !pip install fvcore

# !pip install -qq torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# !pip install -qq git+https://github.com/qubvel/segmentation_models.pytorch
# !pip install -qq timm==0.4.12
# !pip install -qq einops

# !cp ../drive/MyDrive/Coat/coat.py ..
# !cp ../drive/MyDrive/Coat/daformer.py ..
# !cp ../drive/MyDrive/Coat/helper.py ..
# !cp ../drive/MyDrive/Coat/coat_lite_medium_384x384_f9129688.pth ..

"""# config"""

import os
import gc
import sys
import glob

import torch
import torch.nn as nn
import albumentations as A

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import segmentation_models_pytorch as smp
from sklearn.model_selection import StratifiedKFold

import tifffile as tiff
import shutil

torch.backends.cudnn.benchmark = True

fold = 1
nfolds = 4
imsize = 640
train_csv = '../train.csv'
BATCH_SIZE = 4
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
NUM_WORKERS = 1
SEED = 24
TRAIN_PATH = '../640dataset/train/'
MASK_PATH = '../640dataset/masks/'

def set_seed(seed=12):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed(12)

class HuBMAPDataset(torch.utils.data.Dataset):
    def __init__(self, fold=fold, train=True, tfms=None):
        self.train = train
        ids = pd.read_csv(train_csv).id.values
        labels = pd.read_csv(train_csv).organ.values
        kf = StratifiedKFold(n_splits=nfolds,random_state=SEED,shuffle=True)
        ids = (ids[list(kf.split(ids,labels))[fold][0 if train else 1]]).tolist()
        self.fnames = [fname for fname in os.listdir(TRAIN_PATH) if int(fname.split('_')[0]) in ids]
        self.image_size = imsize
        self.tfms = tfms
        
    def img2tensor(self, img,dtype:np.dtype=np.float32):
        if img.ndim==2 : img = np.expand_dims(img,2)
        img = np.transpose(img,(2,0,1)) # C , H , W
        return torch.from_numpy(img.astype(dtype, copy=False))
    
    def __len__(self):
        return len(self.fnames)
    
    def resize(self, img, interp):
        return  cv2.resize(
            img, (self.image_size, self.image_size), interpolation=interp)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(TRAIN_PATH + fname), cv2.COLOR_BGR2RGB)
        mask = cv2.imread((MASK_PATH + fname),cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']
        return self.img2tensor(self.resize(img , cv2.INTER_NEAREST)) , self.img2tensor(self.resize(mask , cv2.INTER_NEAREST))

def transformer(p=1.0):
    return A.Compose([
        A.GridDropout(ratio=0.4, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None, shift_x=0, shift_y=0, 
                     random_offset=True, fill_value=0, mask_fill_value=0, always_apply=False, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Morphology
        A.ShiftScaleRotate(shift_limit=(-0.1, 0.1), scale_limit=(-0.2, 0.2), rotate_limit=(-30, 30), interpolation=1, border_mode=0, value=(0, 0, 0), p=0.4),
        A.GaussNoise(var_limit=(5.0, 50.0), mean=10, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),

        # Color
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5,
                                 brightness_by_max=True, p=0.5),
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
                           val_shift_limit=30, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.5),
            A.PiecewiseAffine(p=0.5),
        ], p=0.5),
    ], p=p)

# ds = HuBMAPDataset(tfms=transformer())
# dl = torch.utils.data.DataLoader(ds,batch_size=64,shuffle=False,num_workers=NUM_WORKERS)
# it = iter(dl)
# imgs,masks = next(it)

# plt.figure(figsize=(16,16))
# for i,(img,mask) in enumerate(zip(imgs,masks)):
#     img = ((img.permute(1,2,0))).numpy().astype(np.uint8)  # H , W , C
#     plt.subplot(8,8,i+1)
#     plt.imshow(img,vmin=0,vmax=255)
#     plt.imshow(mask.squeeze().numpy(), alpha=0.2)
#     plt.axis('off')
#     plt.subplots_adjust(wspace=None, hspace=None)
    
# del ds,dl,imgs,masks

"""# model"""

from coat import *
from daformer import *
from helper import *

class Net(nn.Module):
    
    def __init__(self,
                 encoder=coat_lite_medium,
                 decoder=daformer_conv3x3,
                 encoder_cfg={},
                 decoder_cfg={},
                 ):
        
        super(Net, self).__init__()
        decoder_dim = decoder_cfg.get('decoder_dim', 320)

        self.encoder = encoder

        self.rgb = RGB()

        encoder_dim = self.encoder.embed_dims
        # [64, 128, 320, 512]

        self.decoder = decoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )
        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
            nn.Upsample(scale_factor = 4, mode='bilinear', align_corners=False),
        )

    def forward(self, batch):

        x = self.rgb(batch)

        B, C, H, W = x.shape
        encoder = self.encoder(x)

        last, decoder = self.decoder(encoder)
        logit = self.logit(last)

        output = {}
        probability_from_logit = torch.sigmoid(logit)
        output['probability'] = probability_from_logit

        return output

def init_model():
    encoder = coat_lite_medium()
    checkpoint = './coat_lite_medium_384x384_f9129688.pth'
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    state_dict = checkpoint['model']
    encoder.load_state_dict(state_dict,strict=False)
    
    net = Net(encoder=encoder).cuda()
    
    return net

"""# metric"""

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss,self).__init__()
        self.diceloss = smp.losses.DiceLoss(mode='binary')
        self.binloss = smp.losses.SoftBCEWithLogitsLoss(reduction = 'mean' , smooth_factor = 0.1)

    def forward(self, output, mask):
        dice = self.diceloss(outputs,mask)
        bce = self.binloss(outputs , mask)
        loss = dice * 0.7 + bce * 0.3
        return loss

class DiceCoef(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, y_pred, y_true, smooth=1.):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        
        #Round off y_pred
        y_pred = torch.round((y_pred - y_pred.min()) / (y_pred.max() - y_pred.min()))
        
        intersection = (y_true * y_pred).sum()
        dice = (2.0*intersection + smooth)/(y_true.sum() + y_pred.sum() + smooth)
        
        return dice

def plot_df(df):
    fig,ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].plot(df['Train_loss'])
    ax[0].plot(df['Val_loss'])
    ax[0].legend()
    ax[0].set_title('Loss')
    ax[1].plot(df['Train_Dice'])
    ax[1].plot(df['Val_Dice'])
    ax[1].legend()
    ax[1].set_title('Dice')

"""# train"""

import shutil

# shutil.rmtree('./models')

print(f"Running on device :  {DEVICE}" )
if not os.path.exists("models"):
        os.mkdir("models")
for fold in range(0, 1):

    if not os.path.exists(f"models/fold_{fold}"):
        os.mkdir(f"models/fold_{fold}")
    
    val_losses = []
    losses = []
    train_scores=[]
    val_scores = []
    best_loss = 999
    best_score = 0
    
    ds_train = HuBMAPDataset(fold=fold, train=True, tfms=transformer())
    ds_val = HuBMAPDataset(fold=fold, train=False)
    
    dataloader_train = torch.utils.data.DataLoader(ds_train,batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
    dataloader_val = torch.utils.data.DataLoader(ds_val,batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)
    
    model = init_model().to(DEVICE)
    
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 5e-5}, 
        {'params': model.encoder.parameters(), 'lr': 8e-5},  
    ])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(dataloader_train))
    
    loss_func = CustomLoss()
    dice_coe = DiceCoef()
    
    print(f"######## FOLD: {fold} ##############")
    
    for epoch in range(EPOCHS):


        
        ### Train ###########################################################################################
        
        model.train()
        train_loss = 0
        score = 0
        
        for data in (dataloader_train) :# ,total = len(dataloader_train)):
            optimizer.zero_grad()
            img, mask = data
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
        
            outputs = model(img)['probability']    

            loss = loss_func(outputs, mask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            score += dice_coe(outputs,mask).item()
            
        train_loss /= len(dataloader_train)
        score /= len(dataloader_train)
        losses.append(train_loss)
        train_scores.append(score)
        print(f"FOLD: {fold}, EPOCH: {epoch + 1}, train_loss: {train_loss} , Dice coe : {score} ") #
        
        
        gc.collect()
        torch.cuda.empty_cache()
        
        ### Validation ####################################################################################
        
        model.eval()
        
        with torch.no_grad():
            
            valid_loss = 0
            val_score = 0
            
            for data in dataloader_val:
                
                img, mask = data
                img = img.to(DEVICE)
                mask = mask.to(DEVICE)

                outputs = model(img)['probability']

                loss = loss_func(outputs, mask)
                valid_loss += loss.item()
                val_score += dice_coe(outputs,mask).item()
                
            valid_loss /= len(dataloader_val)
            val_losses.append(valid_loss)
            
            val_score /= len(dataloader_val)
            val_scores.append(val_score)
            
            print(f"FOLD: {fold}, EPOCH: {epoch + 1}, valid_loss: {valid_loss} , Val Dice COE : {val_score}") #
            
            gc.collect()
            torch.cuda.empty_cache()





            
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), f"models/fold_{fold}/FOLD{fold}_best_score.pth")
            print(f"Saved model for best score : FOLD{fold}_best_score.pth")
            
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"models/fold_{fold}/FOLD{fold}_best_loss.pth")
            print(f"Saved model for best loss : FOLD{fold}_best_loss.pth")


    save_path = './output/'
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    shutil.copy(f"models/fold_{fold}/FOLD{fold}_best_score.pth", save_path)
    shutil.copy(f"models/fold_{fold}/FOLD{fold}_best_loss.pth", save_path)

    column_names = ['Train_loss','Val_loss','Train_Dice','Val_Dice']
    df = pd.DataFrame(np.stack([losses,val_losses,train_scores,val_scores],axis=1),columns=column_names)
    print(f" ################# FOLD {fold} #####################")
    plot_df(df)
    plt.show(block=False)
    df.to_csv(f"logs_fold{fold}.csv")

    shutil.copy(f"logs_fold{fold}.csv", save_path)

