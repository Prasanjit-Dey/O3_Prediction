import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn.model_selection import KFold
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import gc
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from google.colab import files
_ = files.upload()

!unzip O3.zip -d "O3"
from IPython.display import clear_output
clear_output(wait=False)
!rm O3.zip

files = pd.read_csv("/content/O3/O3/files.csv")

files.drop("Fold",axis=1,inplace=True)

files

def loadData(df,satdir = "/content/O3/O3/satellite/",gdir = "/content/O3/O3/ground/"):
  X = []
  Y = []
  for i in range(df.shape[0]):
    factor = 46*(6.02214/6.023)*1e2
    sat = np.expand_dims(factor*np.load(os.path.join(satdir,df["SatFile"].iloc[i])),axis=2)
    ground = np.load(os.path.join(gdir,df["GroundFile"].iloc[i])).flatten()
    if not np.isnan(np.sum(sat)) and not np.isnan(np.sum(ground)):
      if not np.std(ground)==0:
        X.append(sat)
        Y.append(ground)
  return np.stack(X,axis=0),np.stack(Y,axis=0)

!pip install torch
!pip install vit-pytorch

#Build the model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision.models import vision_transformer
from vit_pytorch import ViT
from math import ceil

def build_model(X_train):
    #print(X_train.shape)
    # Calculate the image size and channels
    patch_size = 16  # Choose an appropriate patch size
    channels = X_train.shape[-1]
    #print("channels",channels)
    #desired_height = 48
    #desired_width = 48
    #transform = Resize((desired_height, desired_width))
    #X_train_transformed = transform(X_train)

    image_height = X_train.shape[1]  # Use the transformed image height
    image_width = X_train.shape[2]  # Use the transformed image width

    # Adjust the desired height and width to be divisible by the patch size
    resized_height = patch_size * (image_height // patch_size)
    resized_width = patch_size * (image_width // patch_size)

    assert resized_height > 0 and resized_width > 0, "Image dimensions are too small."

    model = ViT(
        image_size=(resized_height, resized_width),
        patch_size=patch_size,
        num_classes=6106,
        dim=64,
        depth=12,
        heads=8,
        mlp_dim=128,
        channels=channels
    )

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    return model, criterion, optimizer

def DataOfFractionSize(df,fraction,n_folds):
  df = df.reset_index(drop=True)
  mask = [False for _ in range(len(df))]

  indices = np.random.choice(list(df.index),int(len(df)*fraction),replace=False)
  for index in indices:
    mask[index]=True
  df = df.loc[mask,:].reset_index(drop=True)

  kf = KFold(n_splits = n_folds,shuffle=True)
  df.loc[:,'Fold'] = -1
  for idx,(trn_,val_) in enumerate(kf.split(df)):
    df.loc[val_,'Fold'] = idx

  return df

import warnings
import numpy as np
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ColorJitter
from torchvision import transforms

def trainFunction(df, n_folds):
    scores_list = []
    rmses = []
    patch_size = 16
    warnings.filterwarnings("ignore", category=UserWarning)

    for fold in range(n_folds):
        train_files = df[df["Fold"] != fold]
        val_files = df[df["Fold"] == fold]

        X_train, Y_train = loadData(train_files)
        X_val, Y_val = loadData(val_files)
        X_train_tensor = torch.from_numpy(X_train)  # Convert X_train to a PyTorch tensor

        desired_height = 64
        desired_width = 80

        # Data augmentation
        transform = transforms.Compose([
            RandomCrop((desired_height, desired_width)),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            Resize((desired_height, desired_width))
        ])

        X_train = transform(torch.Tensor(X_train).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        X_val = transform(torch.Tensor(X_val).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        X_train_tensor = transform(X_train_tensor.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        model, criterion, optimizer = build_model(X_train_tensor)
        train_dataset = torch.utils.data.TensorDataset(X_train.permute(0, 3, 1, 2), torch.Tensor(Y_train))
        val_dataset = torch.utils.data.TensorDataset(X_val.permute(0, 3, 1, 2), torch.Tensor(Y_val))
        train_dataset = torch.utils.data.TensorDataset(X_train.permute(0, 3, 1, 2), torch.Tensor(Y_train))
        val_dataset = torch.utils.data.TensorDataset(X_val.permute(0, 3, 1, 2), torch.Tensor(Y_val))

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)

        for epoch in range(20):
            model.train()
            epoch_train_losses = []
            for inputs, targets in train_loader:
                optimizer.zero_grad()

                resized_inputs = inputs.clone()
                resized_inputs[:, :, :desired_height, :desired_width] = transform(inputs[:, :, :desired_height, :desired_width])

                resized_width = patch_size * (resized_inputs.shape[3] // patch_size)
                resized_inputs = resized_inputs[:, :, :, :resized_width]

                outputs = model(resized_inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                val_outputs = []
                for inputs, targets in val_loader:
                      outputs = model(inputs)
                      val_outputs.append(outputs)
                val_outputs = torch.cat(val_outputs, dim=0)

            rmse = mean_squared_error(Y_val, val_outputs, squared=False)



        rmse1 = mean_squared_error(Y_val, val_outputs, squared=False)
        rmses.append(rmse1)
          #mae = mean_absolute_error(Y_val, val_outputs)
          #maes.append(mae)

          #print("Fold {} RMSE Score: {}".format(fold, rmse))
          #print("Fold {} MAE Score: {}".format(fold, mae))
        del model
        _ = gc.collect()

    print("\nCV RMSE Score: {}".format(np.mean(rmses)))
    return np.mean(rmses)

EXP_PER_SIZE = 20
N_FOLDS = 3
FRACTIONS = np.arange(0.1,1.1,0.1)

frac_vs_rmse = {}
for frac in FRACTIONS:
  print("\nTraining for {} % Dataset\n".format(int(frac*100)))
  exp_rmses = []
  for idx in range(EXP_PER_SIZE):
    print(f"Exp {idx+1}")
    exp_df = DataOfFractionSize(files,frac,N_FOLDS)
    exp_rmses.append(trainFunction(exp_df,N_FOLDS))
  frac_vs_rmse[frac] = exp_rmses

p_data = []
scores = []
for p in FRACTIONS:
  score_ls = frac_vs_rmse[p]
  p_data.extend([int(p*100) for _ in range(len(score_ls))])
  scores.extend(score_ls)
results = pd.DataFrame({"Percentage of Data":p_data,"RMSE Scores":scores})

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 16

fig = plt.figure(figsize=(12, 8))

boxprops = dict(linewidth=1.5, edgecolor='black')
whiskerprops = dict(linewidth=1.5, color='black', solid_capstyle='butt')
capprops = dict(linewidth=1.5, color='black')
medianprops = dict(linewidth=1.5, color='black')
flierprops = dict(marker='o', markersize=5, markerfacecolor='black', markeredgecolor='black')

sns.boxplot(x='Percentage of Data', y='RMSE Scores', data=results, color="lightgreen",
            boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
            medianprops=medianprops, flierprops=flierprops)

plt.show()

fig.savefig("NO2_rmse_data.pdf")

for p in results["Percentage of Data"].unique():
  print("Percentage: ",p)
  temp_df = results[results["Percentage of Data"]==p]
  print("Mean: ",np.mean(temp_df["RMSE Scores"]))
  print("IQR: ",np.percentile(temp_df["RMSE Scores"], 75, interpolation = 'midpoint')-np.percentile(temp_df["RMSE Scores"], 25, interpolation = 'midpoint'))
  print('')