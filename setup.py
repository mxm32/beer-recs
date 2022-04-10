import os
import urllib
import zipfile
import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
ratings = pd.read_csv(data)
ratings.head()

# Data Cleaning

#remove unecessary features
ratings = ratings.drop(ratings.columns[[1,2,4,5,8,9]],axis=1)

#remove rows with NANs ~4.5% of Data
ratings = ratings.dropna(axis=0)

# rename columns
ratings = ratings.rename(columns={'brewery_id': 'breweryID','review_profilename':'reviewer','beer_style':'style','beer_name':'beer_name','beer_abv':'ABV','beer_beerid':'beerID'})

#reorder columns
columns_titles = ["reviewer","review_overall","breweryID","style","beer_name","ABV","beerID"]
ratings=ratings.reindex(columns=columns_titles)

# Creating Dataset for Collaborative Filtering
X = ratings.loc[:,['reviewer','beerID']]
y = ratings.loc[:,'review_overall']

# Creating Dataset for Hybrid
X_h = ratings.loc[:,['reviewer','breweryID','style','ABV','beerID']]
y_h = ratings.loc[:,'review_overall']

#Ordinal Encoding
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# Fit the encoder on training data and transform it.  We can also use it to transform test data
ordinal_cols = ['reviewer']
X[ordinal_cols] = enc.fit_transform(X[ordinal_cols])

X.head()
X['reviewer'] = X['reviewer'].astype(int)
X['beerID'] = X['beerID'].astype(int)

# Split our data into training and test sets
X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0, test_size=0.2)

def prep_dataloaders(X_train,y_train,X_val,y_val,batch_size):
    # Convert training and test data to TensorDatasets
    trainset = TensorDataset(torch.from_numpy(np.array(X_train)).long(),
                            torch.from_numpy(np.array(y_train)).float())
    valset = TensorDataset(torch.from_numpy(np.array(X_val)).long(),
                            torch.from_numpy(np.array(y_val)).float())

    # Create Dataloaders for our training and test data to allow us to iterate over minibatches
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader

batchsize = 64
trainloader,valloader = prep_dataloaders(X_train,y_train,X_val,y_val,batchsize)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloaders = {'train':trainloader, 'val':valloader}
n_users = X.loc[:,'reviewer'].max()+1
n_items = X.loc[:,'beerID'].max()+1
model = NNColabFiltering(n_users,n_items,embedding_dim_users=50, embedding_dim_items=50, n_activations = 100,rating_range=[0.,5.])
criterion = nn.MSELoss()
lr=0.001
n_epochs=10
wd=1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

cost_paths = train_model(model,criterion,optimizer,dataloaders, device,n_epochs, scheduler=None)
