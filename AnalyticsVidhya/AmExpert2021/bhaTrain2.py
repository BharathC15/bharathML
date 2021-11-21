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

from sklearn.metrics import accuracy_score

from tqdm import tqdm

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
            'data' : torch.tensor(data.values,dtype=torch.float32),
            'target' : torch.tensor(target.values,dtype=torch.float32)
        }

        return sample


if __name__=='__main__':
    print('Starting the Program')
    df = pd.read_csv('trainData.csv')
    print(df.head())
    #print(df.info())

    # Extraction of Track columns
    TargetCol = df.columns[df.columns.str.startswith('Target')]
    
    Tensor_dataset = bhaDataset(df)
    
    #print('Sample Dataset')
    #print(Tensor_dataset[0])

    bhaDataloader = DataLoader(Tensor_dataset, batch_size=10, shuffle=True, num_workers=4)

    """
    # Testing of Dataloader
    print('Dataloader')
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['data'].size(),sample_batched['target'].size())
        if i_batch == 2:
            break
    """
    #from sklearn.metrics import accuracy_score
    #print(accuracy_score(Tensor_dataset[0]['target'],Tensor_dataset[1]['target']))
    #print(bha_loss_fn(Tensor_dataset[0]['target'],Tensor_dataset[0]['target']))

    """
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print('Input',input)
    print('Target',target)
    print('Error Result',loss(input,target))
    """

    model = bhaModel(inShape=df.shape[1]-len(TargetCol),outShape=len(TargetCol))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 10
    # load the model on to the computation device
    model.to(device)

    #criterion = accuracy_score
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0

        for i,data in enumerate(bhaDataloader,0):
            inputs, labels = data['data'],data['target']

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 199:
                #print('Epoch :',epoch+1,'items: ',i+1,'Loss',running_loss/200)
                print('Epoch: %d Items: %5d Loss: %.5f '%(epoch+1,i+1,running_loss/200.00))
                running_loss = 0.0