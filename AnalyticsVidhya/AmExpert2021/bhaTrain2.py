# pyTorch
# https://debuggercafe.com/deep-learning-architectures-for-multi-label-classification-using-pytorch/
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class bhaModel(nn.Module):
    def __init__(self,inShape,outShape):
        super().__init__()

        self.inShape = inShape
        self.outShape = outShape

        self.fc1 = nn.Linear(self.inShape, 32) # 12 is the number of features
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256,self.outShape)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        return x

class bhaDataset(Dataset):
    def __init__(self,df, transform=None):
        
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print('Target Col',TargetCol)
        data = self.df.loc[idx, ~self.df.columns.isin(TargetCol)] # Exclude the target columns
        target = self.df.loc[idx, self.df.columns.isin(TargetCol)]

        #print('Data',data.values)
        #print('Target',target.values)

        sample = {
            'data' : torch.tensor(data.values),
            'target' : torch.tensor(target.values)
        }

        return sample

def bha_loss_fn(output,target):
    tempSum = 0
    #print('outputTensor',output)
    #print(len(output))
    for i in range(len(output)):
        tempSum += nn.CrossEntropyLoss()(output[i], target[i])
    return tempSum/len(output)


if __name__=='__main__':
    print('Starting the Program')
    df = pd.read_csv('trainData.csv')
    print(df.head())
    #print(df.info())

    # Extraction of Track columns
    TargetCol = df.columns[df.columns.str.startswith('Target')]
    
    Tensor_dataset = bhaDataset(df)
    
    print('Sample Dataset')
    print(Tensor_dataset[0])

    dataloader = DataLoader(Tensor_dataset, batch_size=4, shuffle=True, num_workers=2)

    """
    # Testing of Dataloader
    print('Dataloader')
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['data'].size(),sample_batched['target'].size())
        if i_batch == 2:
            break
    """
    print(bha_loss_fn(Tensor_dataset[0]['target'],Tensor_dataset[0]['target']))