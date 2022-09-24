import gc
import os
import cv2
import zipfile
import rasterio
import numpy as np
import pandas as pd
from PIL import Image
import tifffile as tiff
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from rasterio.windows import Window
from torch.utils.data import Dataset
import shutil

plt.style.use("Solarize_Light2")

"""### Dataset"""


path_to_zip_file = '/home/ubuntu/binpark/hubmap-organ-segmentation.zip'
directory_to_extract_to = '/home/ubuntu/binpark/'

with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

OUT_TRAIN = '/home/ubuntu/binpark/train.zip'
OUT_MASKS = '/home/ubuntu/binpark/masks.zip'
sz = 640   # the size of tiles
reduce = 3 # reduce the original images by 4 times


MASKS = '/home/ubuntu/binpark/train.csv'
DATA = '/home/ubuntu/binpark/train_images'

# functions to convert encoding to mask and mask to encoding
def enc2mask(mask_rle, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

mask_map = dict(
    kidney=1,
    prostate=2,
    largeintestine=3,
    spleen=4,
    lung=5) #add

df_masks = pd.read_csv(MASKS)[['id', 'organ', 'rle']].set_index('id')#add
df_masks.head()

s_th = 40  # saturation blancking threshold
p_th = 1000*(sz // 640) ** 2 # threshold for the minimum number of pixels


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce, encs=None):
        self.data = rasterio.open(os.path.join(DATA,str(idx)+'.tiff'),num_threads='all_cpus')
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.reduce = reduce
        self.sz = reduce*sz
        self.pad0 = (self.sz - self.shape[0]%self.sz)%self.sz
        self.pad1 = (self.sz - self.shape[1]%self.sz)%self.sz
        self.n0max = (self.shape[0] + self.pad0)//self.sz
        self.n1max = (self.shape[1] + self.pad1)//self.sz
        self.mask = enc2mask(encs,(self.shape[1],self.shape[0])) if encs is not None else None
        
    def __len__(self):
        return self.n0max*self.n1max
    
    def __getitem__(self, idx):
        n0,n1 = idx//self.n1max, idx%self.n1max
        # x0,y0 - are the coordinates of the lower left corner of the tile in the image
        # negative numbers correspond to padding (which must not be loaded)
        x0,y0 = -self.pad0//2 + n0*self.sz, -self.pad1//2 + n1*self.sz

        # make sure that the region to read is within the image
        p00,p01 = max(0,x0), min(x0+self.sz,self.shape[0])
        p10,p11 = max(0,y0), min(y0+self.sz,self.shape[1])
        img = np.zeros((self.sz,self.sz,3),np.uint8)
        mask = np.zeros((self.sz,self.sz),np.uint8)
        # mapping the loade region to the tile
        if self.data.count == 3:
            img[(p00-x0):(p01-x0),(p10-y0):(p11-y0)] = np.moveaxis(self.data.read([1,2,3],
                window=Window.from_slices((p00,p01),(p10,p11))), 0, -1)
        else:
            for i,layer in enumerate(self.layers):
                img[(p00-x0):(p01-x0),(p10-y0):(p11-y0),i] =\
                  layer.read(1,window=Window.from_slices((p00,p01),(p10,p11)))
        if self.mask is not None: mask[(p00-x0):(p01-x0),(p10-y0):(p11-y0)] = self.mask[p00:p01,p10:p11]
        
        if self.reduce != 1:
            img = cv2.resize(img,(self.sz//reduce,self.sz//reduce),
                             interpolation = cv2.INTER_AREA)
            mask = cv2.resize(mask,(self.sz//reduce,self.sz//reduce),
                             interpolation = cv2.INTER_NEAREST)
        #check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        #return -1 for empty images
        return img, mask, (-1 if (s>s_th).sum() <= p_th or img.sum() <= p_th else idx)

x_tot,x2_tot = [],[]
mask_ = [] #add
with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out,\
 zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
    for index,(organ, encs) in tqdm(df_masks.iterrows(),total=len(df_masks)):#add
        ds = HuBMAPDataset(index,encs=encs)
        for i in range(len(ds)):    
            im,m,idx = ds[i]
            if idx < 0: continue

            x_tot.append((im/255.0).reshape(-1,3).mean(0))
            x2_tot.append(((im/255.0)**2).reshape(-1,3).mean(0))
            m = mask_map[organ]*m ######### #add
            for e in np.unique(m): #add
                mask_.append(e) #add
            im = cv2.imencode('.png',cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1]
            img_out.writestr(f'{index}_{idx:04d}.png', im)
            m = cv2.imencode('.png',m)[1]
            mask_out.writestr(f'{index}_{idx:04d}.png', m)

        
#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', img_std)
print('classes of mask', set(mask_)) #add
print(img_avr*255, img_std*255)


"""### Dataset unzip"""

os.makedirs('/home/ubuntu/binpark/mmsegmentation640_reduce3/train', exist_ok=True)
os.makedirs('/home/ubuntu/binpark/mmsegmentation640_reduce3/masks', exist_ok=True)


train_zip_file = '/home/ubuntu/binpark/train.zip'
train_extract = '/home/ubuntu/binpark/mmsegmentation640_reduce3/train'

with zipfile.ZipFile(train_zip_file, 'r') as zip_ref:
    zip_ref.extractall(train_extract)


masks_zip_file = '/home/ubuntu/binpark/masks.zip'
masks_extract = '/home/ubuntu/binpark/mmsegmentation640_reduce3/masks/'

with zipfile.ZipFile(masks_zip_file, 'r') as zip_ref:
    zip_ref.extractall(masks_extract)


"""#2

### 폴더복사
"""

os.makedirs('/home/ubuntu/binpark/mmseg_data640_3/splits', exist_ok=True)

# #덮어쓰기용
# from distutils.dir_util import copy_tree
# copy_tree('/home/ubuntu/binpark/mmsegmentation768x768/masks', '/home/ubuntu/binpark/mmseg_data640_3/masks')
# copy_tree('/home/ubuntu/binpark/mmsegmentation768x768/train', '/home/ubuntu/binpark/mmseg_data640_3/train')

shutil.move('/home/ubuntu/binpark/mmsegmentation640_reduce3/train', '/home/ubuntu/binpark/mmseg_data640_3')
shutil.move('/home/ubuntu/binpark/mmsegmentation640_reduce3/masks', '/home/ubuntu/binpark/mmseg_data640_3')




from glob import glob
import numpy as np
import cv2
import os
from sklearn.model_selection import StratifiedKFold

Fold = 5 # 10개의 fold(그냥 train data같음), 10개의 valid로 나뉨

all_mask_files = glob("/home/ubuntu/binpark/mmseg_data640_3/masks/*")
masks = []
num_mask = np.zeros((6,Fold))

for i in range(len(all_mask_files)):
    mask = cv2.imread(all_mask_files[i])
    masks.append(mask.max())

# 이해 못함
split = list(StratifiedKFold(n_splits=Fold, shuffle=True, random_state=7).split(all_mask_files, masks))

for fold, (train_idx, valid_idx) in enumerate(split):
    for i in valid_idx:
        num_mask[masks[i]]+=1
    with open(f"/home/ubuntu/binpark/mmseg_data640_3/splits/fold_{fold}.txt", "w") as f:
        for idx in train_idx:
            f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
    with open(f"/home/ubuntu/binpark/mmseg_data640_3/splits/valid_{fold}.txt", "w") as f:
        for idx in valid_idx:
            f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
print(num_mask)