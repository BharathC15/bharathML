#%% 

import numpy as np
import pandas as pd

# %%

testData = pd.read_csv('train_go05W65.csv')
print(testData.head())
print(testData.info())

# %%
print(testData.mean(numeric_only=True))

# %%

import functools
import itertools

print('\nUnique Gender',testData['Gender'].unique())
print('\nUnique city category',testData['City_Category'].unique())
print('\nUnique customer category',testData['Customer_Category'].unique())
print('\nUnique Holding B1',testData['Product_Holding_B1'].unique())
print('\nUnique Holding B2',testData['Product_Holding_B2'].unique())


# %%
uniqueProduct = []
for i in testData['Product_Holding_B1']:
    for j in i.replace('[',"").replace(']',"").replace('\'',"").replace(" ","").split(','):
        uniqueProduct.append(j)
uniqueProduct = list(set(uniqueProduct))
print('Unique B1 :',uniqueProduct)

# %%
print(testData.head())

# %%
