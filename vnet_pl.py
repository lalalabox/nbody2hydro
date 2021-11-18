import os
from datetime import datetime
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchio as tio
import monai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %load_ext tensorboard

'''check GPU'''
# let's see how many GPU we have?
ngpu = torch.cuda.device_count()
print(f"We have {ngpu} GPU(s) on this machine")

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("驱动为：",device)
# print("GPU型号： ",torch.cuda.get_device_name(0))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

'''DATA'''
class nbody2hydroDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size, train_val_ratio):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.train_val_ratio = train_val_ratio
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
    def prepare_data(self):
        '''load data
        1. get average value in each cell
        2. convert nan -> 0
        3. convert mass [log10]
        '''
        def _sort_dict(d):
            return OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        # load data
        INPUT = {
            'dm_velx': 'TNG-50-hydro-z0-dm-mesh-vx.npy',
            'dm_vely': 'TNG-50-hydro-z0-dm-mesh-vy.npy',
            'dm_velz': 'TNG-50-hydro-z0-dm-mesh-vz.npy',
            'dm_mass': 'TNG-50-hydro-z0-dm-mesh-v-num.npy'
        }
        OUTPUT = {
            'sub_mass': 'TNG-50-hydro-z0-subhalo-SubhaloMass.npy'
        }
        NUMS = {
            'dm': 'TNG-50-hydro-z0-dm-mesh-v-num.npy',
            'sub': 'TNG-50-hydro-z0-subhalo-Position.npy'
        }
        # sort dict
        INPUT = _sort_dict(INPUT)
        OUTPUT = _sort_dict(OUTPUT)
        NUMS = _sort_dict(NUMS)

        for key in NUMS.keys():
            exec('{} = np.load("{}")'.format(key+'_nums',self.dataset_dir+NUMS[key]))
        # pre-processing
        # divided by nums in each cell [get average num in each voxel]
        for key in INPUT.keys():
            exec('{} = np.load("{}")'.format(key,self.dataset_dir+INPUT[key]))
            # mass [log10]
            if key=='dm_mass':
                exec('{} = np.log10({}+1)'.format(key,key))
                pass
            # velocity [nan -> 0]
            else:
                exec('{} = np.divide({}, {})'.format(key,key,'dm_nums'))  # divide
                exec('np.nan_to_num({}, copy=False,nan=0)'.format(key))  # nan -> 0
            pass
        for key in OUTPUT.keys():
            exec('{} = np.load("{}")'.format(key,self.dataset_dir+OUTPUT[key]))
            if key == 'sub_mass':
                exec('{} = np.divide({}, {})'.format(key,key,'sub_nums'))  # divide
                exec('np.nan_to_num({}, copy=False,nan=0)'.format(key))  # nan -> 0
                exec('{} = np.log10({}+1)'.format(key,key))  # log10
            pass

        ## check nan or inf
        #for key in INPUT.keys():
        #    #exec('print({}.shape)'.format(key))
        #    exec('print("{}: ", {}.shape)'.format(key,key))
        #    exec('print("min: ",np.min({}))'.format(key))
        #    exec('print("max: ",np.max({}))'.format(key))
        #    pass
        #for key in OUTPUT.keys():
        #    exec('print("{}: ", {}.shape)'.format(key,key))
        #    exec('print("min: ",np.min({}))'.format(key))
        #    exec('print("max: ",np.max({}))'.format(key))
        #    pass
        # numpy to torch tensor
        def _arr2tensor(dict_keys):
            vals = []
            for key in dict_keys:
                exec('vals.append(torch.from_numpy({}))'.format(key))
                pass
            return torch.stack(vals)
        input_3D = _arr2tensor(INPUT.keys())
        output_3D = _arr2tensor(OUTPUT.keys())
        #print(input_3D.shape,output_3D.shape)
        '''normalize
        可选择 torchio.transforms.HistogramStandardization
        首先直接尝试 rescale
        '''
        rescale11 = tio.RescaleIntensity(out_min_max=(-1, 1))
        rescale01 = tio.RescaleIntensity(out_min_max=(0, 1))

        # vel -> [-1, 1]
        input_3D[1:] = rescale11(input_3D[1:])
        # mass -> [0,1]
        input_3D[:1] = rescale11(input_3D[:1])
        output_3D[:1] = rescale11(output_3D[:1])
        '''augmentation
        original: 10x10x10 = 1000
        using random center, we can get more samples!
        '''
        NUM_SAMPLES = 10000

        transform = monai.transforms.RandSpatialCropSamples(
            roi_size = [64,64,64], 
            num_samples = NUM_SAMPLES,
            random_size=False
        )

        _inputs_3D = transform(input_3D)
        _outputs_3D = transform(output_3D)
        self.subjects = [
            tio.Subject(
                inputs = tio.ScalarImage(tensor = _in),
                outputs=tio.ScalarImage(tensor = _out)
            )
            for _in, _out in zip(_inputs_3D, _outputs_3D)
        ]
        nums_test = int(0.2*len(subjects))
        self.test_subjects = self.subjects[:nums_test]
        self.subjects = self.subjects[nums_test:]

    def setup(self, stage=None):
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)

        self.transform = tio.Compose([tio.EnsureShapeMultiple(8)])

        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.transform)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size)
    
    
HOME = os.path.expandvars('$HOME')
data = nbody2hydroDataModule(
    dataset_dir=HOME+'/code/nbody2hydro/tmp_save/',
    batch_size=8,
    train_val_ratio=0.8,
)
data.prepare_data()
data.setup()
print('Training:  ', len(data.train_set))
print('Validation: ', len(data.val_set))
print('Test:      ', len(data.test_set))

'''MODEL'''
class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, x):
        y_hat = self.net(x)
        return y_hat
    
    def prepare_batch(self, batch):
        return batch['inputs'][tio.DATA], batch['outputs'][tio.DATA]
    
    #def infer_batch(self, batch):
    #    x, y = self.prepare_batch(batch)
    #    y_hat = self.net(x)
    #    return y_hat, y

    def training_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        y_hat = self(x)
        #y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        y_hat = self(x)
        #y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        y_hat = self(x)
        #y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        print(f"test loss is {loss}")
        return loss
    
unet = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=4,
    out_channels=1,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
)

model = Model(
    net=unet,
    criterion=nn.MSELoss(), #F.mse_loss() #monai.losses.DiceCELoss(softmax=True),
    learning_rate=1e-3,
    optimizer_class=torch.optim.AdamW,
)
early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss',
)

# setting trainer
trainer = pl.Trainer(
    gpus=1,
    precision=16,
    callbacks=[early_stopping],
)
trainer.logger._default_hp_metric = False

'''TRAIN'''
start = datetime.now()
print('Training started at', start)
trainer.fit(model=model, datamodule=data)
print('Training duration:', datetime.now() - start)