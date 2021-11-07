# https://www.kaggle.com/raqhea/digit-classification-with-a-linear-model-pytorch?scriptVersionId=78938799

#%%
import torch
from torch import nn

from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import os


# %%
# Importing the Dataset

train_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')
train_data.head()

# %%
# convert data frames to numpy arrays
X_train, Y_train = train_data.iloc[:,1:].values, train_data.iloc[:,0].values

# split the dataset
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = .25)
x_test = test_data.values # we don't have targets in the test data

# plot few samples from the training data 
fig, axs = plt.subplots(1, 3)
idxs = np.random.randint(0, x_train.shape[0], 3)
for i, idx in enumerate(idxs):
    axs[i].set_title(y_train[idx])
    axs[i].imshow(x_train[idx].reshape(28, 28), cmap = 'gray')
plt.show()

#%%
BATCH_SIZE = 256
LR = 0.001
N_EPOCHS = 50
N_CLASSES = len(set(y_train)) # 10
# %%
# standardizing the data

train_mean = x_train.mean()
train_std = x_train.std()

# we will standardize the validation and test set using 
# the mean and standard deviation of training set

x_train = (x_train - train_mean) / train_std
x_val = (x_val - train_mean) / train_std
x_test = (x_test - train_mean) / train_std
# convert everything to torch tensors
x_train = torch.from_numpy(x_train).type(torch.float32)
x_val = torch.from_numpy(x_val).type(torch.float32)
x_test = torch.from_numpy(x_test).type(torch.float32)
y_train = torch.from_numpy(y_train)
y_val = torch.from_numpy(y_val)

# build the datasets and dataloaders
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
# 
train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle = False)
# %%
class LinearModel(nn.Module):
    """ Basic linear model with hidden layers """
    def __init__(self, n_input: int = 784, hidden_nodes = [128], n_output: int = N_CLASSES):
        super().__init__()
        if type(hidden_nodes) == int:
            hidden_nodes = [hidden_nodes]
        # concatenate the number of inputs, outputs and all the hidden nodes list         
        # to automatically build a linear model using ModuleList
        num_nodes = [n_input] + hidden_nodes + [n_output]

        
        # let's start
        # we can group sequential Linear->ReLU models together to build the model automatically
        module_list = nn.ModuleList([
            nn.Sequential(
            nn.Linear(num_nodes[i], num_nodes[i+1]), nn.ReLU()
            )
            for i in range(len(num_nodes)-2)
        ]) # the last layer will have num_nodes[-3] input nodes and num_nodes[-2] output nodes
        
        
        
        self.linear = nn.Sequential(*module_list) # pass the modules in the module list to Sequential model
        # since the last layer in `linear` will have number of num_nodes[-2] output nodes,
        # we will set the input nodes `num_nodes[-2]` in the classifier 
        # we also concatenated `n_output` with `num_nodes` and 
        # last element of `num_nodes` is `n_output`
        # so we will set the number of output nodes as `n_output` by passing `num_nodes[-1]`
        self.classifier = nn.Linear(num_nodes[-2], num_nodes[-1])
        
    def configure(self, optimizer, loss_fn):
        """ Simple function to set the optimizer and loss function """
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def forward(self, x):
        x = self.linear(x) # output shape: Nx(n_hidden)
        outputs = self.classifier(x) # output shape: Nx(n_output)
        return outputs
        
    def train_step(self, x_batch, y_batch):
        # feed data to the network
        out = self.forward(x_batch)
        # calculate the losses
        loss = self.loss_fn(out, y_batch)
        # backward propagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad() # we zero out the gradients when we're done, they will accumulate if we don't.
        # finally return the loss
        return loss.item()
    
    def val_step(self, x_batch, y_batch):
        with torch.no_grad():
            out = self.forward(x_batch)
            loss = self.loss_fn(out, y_batch)
        return loss.item()

    def predict(self, x):
        """ Function to make predictions, just returns the index of the maximums in model outputs """
        model_outputs = self.forward(x)
        preds = model_outputs.argmax(1)
        return preds

        
# %%
model = LinearModel(hidden_nodes = [256,256])
print(model)
del model


# %%
def train(model, train_loader, validation_loader, n_epochs = 20):
    train_losses = np.zeros(n_epochs)
    val_losses = np.zeros(n_epochs)
    for i in range(n_epochs):
        # define the running losses for the epoch
        train_batch_loss = 0
        val_batch_loss = 0
        
        for x, y in train_loader:
            loss = model.train_step(x, y)
            # CrossEntropyLoss is reducing the calculated loss by mean
            # So every single time we add up the average loss by batches
            # we also need to divide it by number of batches because of this
            train_batch_loss += loss
        
        model.eval()
        for x, y in val_loader:
            val_loss = model.val_step(x, y)
            val_batch_loss += val_loss 
        model.train()
        
        # divide it by the number of the batches per dataset phase
        # to get the average losses
        train_losses[i] = train_batch_loss / len(train_loader)
        val_losses[i] = val_batch_loss / len(validation_loader)
        
        if (i+1)%10==0: # we will log the average losses per epoch
            print(f"Epoch {i+1}/{n_epochs} | Avg. training loss: {train_losses[i]:.4f} | Avg. validation loss: {val_losses[i]:.4f}")
    return train_losses, val_losses


# %%
# we can finally define the model and train it
model = LinearModel(hidden_nodes = [256, 256])
print(model)
# define the loss function and optimizer, then pass them to `configure` method of the model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
model.configure(optimizer, loss_fn)


train_losses, val_losses = train(model, train_loader, val_loader, n_epochs = N_EPOCHS)

plt.title('Losses')
plt.plot(train_losses, label = 'training loss')
plt.plot(val_losses, label = 'validation loss')
plt.legend()
plt.show()


# %%
# make predictions using validation data
model.eval()
preds = model.predict(x_val)
model.train()
preds = preds.tolist()

# let's take a look of the model's performance:
print(classification_report(preds, y_val))


# %%
# make predictions using test data and prepare the submission dataframe
model.eval()
preds = model.predict(x_test)
preds = preds.tolist()

submission = pd.read_csv('sample_submission.csv')
submission['Label'] = preds
submission.to_csv('submission.csv', index = None)
# %%
pd.read_csv('submission.csv')
# %%

# %%
