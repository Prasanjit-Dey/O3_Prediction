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

files

def pearson(y_true,y_pred):
  if len(y_true.shape)!=1:
    true = []
    for i in range(y_true.shape[0]):
      true.extend(y_true[i])
    pred = []
    for i in range(y_pred.shape[0]):
      pred.extend(y_pred[i])
  else:
    true=y_true
    pred=y_pred
  return np.mean((np.array(true)-np.mean(true))*(np.array(pred)-np.mean(pred)))/(np.std(np.array(true))*np.std(np.array(pred)))

def pearsonCorrAvgDays(true, pred):
    assert true.shape == pred.shape, "true and pred must have the same shape, found {} and {}".format(true.shape, pred.shape)
    scores = []
    for i in range(true.shape[0]):
        scores.append(np.corrcoef(true[i], pred[i])[0, 1])
    return np.mean(scores), scores

def pearsonCorrAvgPixels(true,pred):
  scores = []
  for i in range(true.shape[1]):
    scores.append(pearson(true[:,i],pred[:,i]))
  return np.mean(scores),scores

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, criterion, optimizer

import warnings
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD
from vit_pytorch import ViT
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ColorJitter
from torchvision import transforms

# Model training
scores_list = []
rmses = []
maes = []
patch_size = 16
warnings.filterwarnings("ignore", category=UserWarning)

for fold in range(5):
    print("\nFold {}\n".format(fold))
    train_files = files[files["Fold"] != fold]
    val_files = files[files["Fold"] == fold]

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
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)  # Example of using SGD optimizer

    if fold == 0:
        #print(model)
        pass

    train_dataset = torch.utils.data.TensorDataset(X_train.permute(0, 3, 1, 2), torch.Tensor(Y_train))
    val_dataset = torch.utils.data.TensorDataset(X_val.permute(0, 3, 1, 2), torch.Tensor(Y_val))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    patience = 5  # Number of epochs to wait for improvement
    best_val_loss = np.inf
    num_epochs_without_improvement = 0

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

    for epoch in range(30):
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



        # Update learning rate
        lr_scheduler.step(rmse)

        # Check for early stopping
        if rmse < best_val_loss:
            best_val_loss = rmse
            num_epochs_without_improvement = 0
        else:
            num_epochs_without_improvement += 1
            if num_epochs_without_improvement == patience:
                print("Early stopping criteria met. Stopping training.")
                break

    rmse1 = mean_squared_error(Y_val, val_outputs, squared=False)
    rmses.append(rmse1)
    mae = mean_absolute_error(Y_val, val_outputs)
    maes.append(mae)

    print("Fold {} RMSE Score: {}".format(fold, rmse))
    print("Fold {} MAE Score: {}".format(fold, mae))
    s, ls = pearsonCorrAvgDays(Y_val, val_outputs)
    print("Fold {} Pearson coeff avg over days: {}".format(fold, np.mean([i for i in ls if not pd.isnull(i)])))
    scores_list.append(ls)
    if fold != 4:
        del model
        _ = gc.collect()

print("\nCV RMSE Score: {}".format(np.mean(rmses)))
print("\nCV MAE Score: {}".format(np.mean(maes)))

"""Calculate daily data"""

print(X_train.shape)
print(X_val.shape)
inp = X_val[0][:,:,0]
print(inp.shape)
#true = Y_val[10].reshape((X_train[0].shape[0],X_train[0].shape[1]))
true = Y_val[0].reshape((71,86))
print(true.shape)
pred = val_outputs[0].reshape((71,86))
print(pred.shape)
#pred = val_outputs.view(X_val[10].shape[1], X_val[10].shape[2])

"""Daily true data"""

import folium
import numpy as np
import branca.colormap as cm

# Create a map centered on Ireland
m = folium.Map(location=[53.349805, -6.26031], zoom_start=7)

ireland_bound = [-10.3401, -6.022 ,51.828500000000005 , 55.422] #Irealand Extent
llcrn,urcrn = (-10.3401, 51.828500000000005),(-6.022, 55.422) # Ireland bounds

# Create latitude and longitude grids
num_cols = int((urcrn[0] - llcrn[0]) // 0.05)
num_rows = int((urcrn[1] - llcrn[1]) // 0.05)
lat_coords = np.linspace(llcrn[1], urcrn[1], num_rows + 1)
lon_coords = np.linspace(llcrn[0], urcrn[0], num_cols + 1)
lat, lon = np.meshgrid(lon_coords, lat_coords)


true = np.array(true)

# Initialize a map
# m = folium.Map(location=[53.0, -7.5], zoom_start=7, tiles='Stamen Terrain')

# Convert the numpy array to a GeoJSON-like structure
features = []
for i in range(num_rows):
    for j in range(num_cols):
        polygon = [
            [lat[i, j], lon[i, j]],
            [lat[i + 1, j], lon[i + 1, j]],
            [lat[i + 1, j + 1], lon[i + 1, j + 1]],
            [lat[i, j + 1], lon[i, j + 1]],
            [lat[i, j], lon[i, j]]
        ]
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            },
            'properties': {
                'value': true[i, j]  # Use the corresponding value from the 'true' array
            }
        }
        features.append(feature)

# Define a color scale for the 'true' values
vmin = np.min(true)
vmax = np.max(true)
colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'], vmin=vmin, vmax=vmax)

# Add the GeoJSON layer to the map and set fill color based on 'true' values
folium.GeoJson(
    {
        'type': 'FeatureCollection',
        'features': features
    },
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['value']),
        'color': 'none',
        'fillOpacity': 0.6
    }
).add_to(m)

# Create a color bar legend with values
legend_html = """
    <div style="
        position: fixed;
        bottom: 360px; left: 520px; width: 70px; height: 134px;
        background-color: white;
        z-index: 9999; font-size: 12px;
        border: 2px solid grey; padding: 2px;
    ">
"""

num_values = 5  # Number of values to display in the color bar legend
value_range = np.linspace(vmin, vmax, num_values)

for value in value_range:
    color = colormap(value)
    label = f'{int(value)}'  # Format the value as desired
    legend_html += f'<p><span style="background-color:{color};">&nbsp;&nbsp;&nbsp;&nbsp;</span> {label}</p>'

legend_html += '</div>'

# Add the color bar legend to the map
m.get_root().html.add_child(folium.Element(legend_html))

# Define the number of rows and columns in the grid for markers
num_rows_markers = 3
num_cols_markers = 3

# Define the latitude and longitude bounds for the grid for markers
llcrn_markers = [51.7665, -10.3401]
urcrn_markers = [55.422, -6.022]

# Calculate the step size for latitude and longitude for markers
lat_step_markers = (urcrn_markers[0] - llcrn_markers[0]) / num_rows_markers
lon_step_markers = (urcrn_markers[1] - llcrn_markers[1]) / num_cols_markers

# Add markers with both DivIcon and Icon inside the rectangle
for i in range(num_rows_markers + 1):
    for j in range(num_cols_markers + 1):
        lat_marker = llcrn_markers[0] + i * lat_step_markers
        lon_marker = llcrn_markers[1] + j * lon_step_markers

        # DivIcon marker
        div_icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.DivIcon(html=f'<div style="font-size: 10pt;">({lat_marker:.4f}, {lon_marker:.4f})</div>')
        )
        div_icon_marker.add_to(m)

        # Icon marker
        icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.Icon(color='green')
        )
        icon_marker.add_to(m)

# Add a rectangle to the map
rectangle_points = [
    [llcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], llcrn_markers[1]],  # Close the rectangle
]

folium.Polygon(
    locations=rectangle_points,
    color='black',  # Line color of the rectangle
    fill=True,
    fill_color='none',  # Fill color of the rectangle
    fill_opacity=0.4,  # Opacity of the fill color
    weight=1.5,  # Width of the rectangle border
).add_to(m)

# Save the map to an HTML file
m.save(f'True_Day_1.html')
m

"""Daily predicted data"""

import folium
import numpy as np
import branca.colormap as cm

# Create a map centered on Ireland
m = folium.Map(location=[53.349805, -6.26031], zoom_start=7)

ireland_bound = [-10.3401, -6.022 ,51.828500000000005 , 55.422] #Irealand Extent
llcrn,urcrn = (-10.3401, 51.828500000000005),(-6.022, 55.422) # Ireland bounds

# Create latitude and longitude grids
num_cols = int((urcrn[0] - llcrn[0]) // 0.05)
num_rows = int((urcrn[1] - llcrn[1]) // 0.05)
lat_coords = np.linspace(llcrn[1], urcrn[1], num_rows + 1)
lon_coords = np.linspace(llcrn[0], urcrn[0], num_cols + 1)
lat, lon = np.meshgrid(lon_coords, lat_coords)


pred = np.array(pred).astype(np.float64)

# Initialize a map
# m = folium.Map(location=[53.0, -7.5], zoom_start=7, tiles='Stamen Terrain')

# Convert the numpy array to a GeoJSON-like structure
features = []
for i in range(num_rows):
    for j in range(num_cols):
        polygon = [
            [lat[i, j], lon[i, j]],
            [lat[i + 1, j], lon[i + 1, j]],
            [lat[i + 1, j + 1], lon[i + 1, j + 1]],
            [lat[i, j + 1], lon[i, j + 1]],
            [lat[i, j], lon[i, j]]
        ]
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            },
            'properties': {
                'value': pred[i, j]  # Use the corresponding value from the 'true' array
            }
        }
        features.append(feature)

# Define a color scale for the 'true' values
vmin = np.min(pred)
vmax = np.max(pred)
colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'], vmin=vmin, vmax=vmax)

# Add the GeoJSON layer to the map and set fill color based on 'true' values
folium.GeoJson(
    {
        'type': 'FeatureCollection',
        'features': features
    },
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['value']),
        'color': 'none',
        'fillOpacity': 0.6
    }
).add_to(m)

# Create a color bar legend with values
legend_html = """
    <div style="
        position: fixed;
        bottom: 360px; left: 520px; width: 70px; height: 134px;
        background-color: white;
        z-index: 9999; font-size: 12px;
        border: 2px solid grey; padding: 2px;
    ">
"""

num_values = 5  # Number of values to display in the color bar legend
value_range = np.linspace(vmin, vmax, num_values)

for value in value_range:
    color = colormap(value)
    label = f'{int(value)}'  # Format the value as desired
    legend_html += f'<p><span style="background-color:{color};">&nbsp;&nbsp;&nbsp;&nbsp;</span> {label}</p>'

legend_html += '</div>'

# Add the color bar legend to the map
m.get_root().html.add_child(folium.Element(legend_html))

# Define the number of rows and columns in the grid for markers
num_rows_markers = 3
num_cols_markers = 3

# Define the latitude and longitude bounds for the grid for markers
llcrn_markers = [51.7665, -10.3401]
urcrn_markers = [55.422, -6.022]

# Calculate the step size for latitude and longitude for markers
lat_step_markers = (urcrn_markers[0] - llcrn_markers[0]) / num_rows_markers
lon_step_markers = (urcrn_markers[1] - llcrn_markers[1]) / num_cols_markers

# Add markers with both DivIcon and Icon inside the rectangle
for i in range(num_rows_markers + 1):
    for j in range(num_cols_markers + 1):
        lat_marker = llcrn_markers[0] + i * lat_step_markers
        lon_marker = llcrn_markers[1] + j * lon_step_markers

        # DivIcon marker
        div_icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.DivIcon(html=f'<div style="font-size: 10pt;">({lat_marker:.4f}, {lon_marker:.4f})</div>')
        )
        div_icon_marker.add_to(m)

        # Icon marker
        icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.Icon(color='green')
        )
        icon_marker.add_to(m)

# Add a rectangle to the map
rectangle_points = [
    [llcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], llcrn_markers[1]],  # Close the rectangle
]

folium.Polygon(
    locations=rectangle_points,
    color='black',  # Line color of the rectangle
    fill=True,
    fill_color='none',  # Fill color of the rectangle
    fill_opacity=0.4,  # Opacity of the fill color
    weight=1.5,  # Width of the rectangle border
).add_to(m)

# Save the map to an HTML file
m.save(f'pred_Day_1.html')
m

"""Calculate weekly data"""

# Initialize variables to store cumulative values
total_inp = np.zeros_like(X_val[0][:, :, 0])
total_true = np.zeros((71, 86))
total_pred = np.zeros((71, 86))

# Loop through the weeks
for week in range(0, 7):
    # Extract data for the current week
    inp_week = X_val[week][:, :, 0]
    true_week = Y_val[week].reshape((71, 86))
    pred_week = val_outputs[week].reshape((71, 86))
    #print(pred_week)
    # Accumulate values element-wise
    np.add(total_inp, inp_week, out=total_inp)
    np.add(total_true, true_week, out=total_true)
    np.add(total_pred, pred_week, out=total_pred)

# Calculate average values
avg_inp = total_inp / 7  # Assuming you have data for 7 weeks
avg_true = total_true / 7
avg_pred = total_pred / 7
#print(avg_pred)

"""Montly true data"""

import folium
import numpy as np
import branca.colormap as cm

# Create a map centered on Ireland
m = folium.Map(location=[53.349805, -6.26031], zoom_start=7)

ireland_bound = [-10.3401, -6.022 ,51.828500000000005 , 55.422] #Irealand Extent
llcrn,urcrn = (-10.3401, 51.828500000000005),(-6.022, 55.422) # Ireland bounds

# Create latitude and longitude grids
num_cols = int((urcrn[0] - llcrn[0]) // 0.05)
num_rows = int((urcrn[1] - llcrn[1]) // 0.05)
lat_coords = np.linspace(llcrn[1], urcrn[1], num_rows + 1)
lon_coords = np.linspace(llcrn[0], urcrn[0], num_cols + 1)
lat, lon = np.meshgrid(lon_coords, lat_coords)


avg_true = np.array(avg_true)

# Initialize a map
# m = folium.Map(location=[53.0, -7.5], zoom_start=7, tiles='Stamen Terrain')

# Convert the numpy array to a GeoJSON-like structure
features = []
for i in range(num_rows):
    for j in range(num_cols):
        polygon = [
            [lat[i, j], lon[i, j]],
            [lat[i + 1, j], lon[i + 1, j]],
            [lat[i + 1, j + 1], lon[i + 1, j + 1]],
            [lat[i, j + 1], lon[i, j + 1]],
            [lat[i, j], lon[i, j]]
        ]
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            },
            'properties': {
                'value': avg_true[i, j]  # Use the corresponding value from the 'true' array
            }
        }
        features.append(feature)

# Define a color scale for the 'true' values
vmin = np.min(avg_true)
vmax = np.max(avg_true)
colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'], vmin=vmin, vmax=vmax)

# Add the GeoJSON layer to the map and set fill color based on 'true' values
folium.GeoJson(
    {
        'type': 'FeatureCollection',
        'features': features
    },
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['value']),
        'color': 'none',
        'fillOpacity': 0.6
    }
).add_to(m)

# Create a color bar legend with values
legend_html = """
    <div style="
        position: fixed;
        bottom: 360px; left: 520px; width: 70px; height: 134px;
        background-color: white;
        z-index: 9999; font-size: 12px;
        border: 2px solid grey; padding: 2px;
    ">
"""

num_values = 5  # Number of values to display in the color bar legend
value_range = np.linspace(vmin, vmax, num_values)

for value in value_range:
    color = colormap(value)
    label = f'{int(value)}'  # Format the value as desired
    legend_html += f'<p><span style="background-color:{color};">&nbsp;&nbsp;&nbsp;&nbsp;</span> {label}</p>'

legend_html += '</div>'

# Add the color bar legend to the map
m.get_root().html.add_child(folium.Element(legend_html))

# Define the number of rows and columns in the grid for markers
num_rows_markers = 3
num_cols_markers = 3

# Define the latitude and longitude bounds for the grid for markers
llcrn_markers = [51.7665, -10.3401]
urcrn_markers = [55.422, -6.022]

# Calculate the step size for latitude and longitude for markers
lat_step_markers = (urcrn_markers[0] - llcrn_markers[0]) / num_rows_markers
lon_step_markers = (urcrn_markers[1] - llcrn_markers[1]) / num_cols_markers

# Add markers with both DivIcon and Icon inside the rectangle
for i in range(num_rows_markers + 1):
    for j in range(num_cols_markers + 1):
        lat_marker = llcrn_markers[0] + i * lat_step_markers
        lon_marker = llcrn_markers[1] + j * lon_step_markers

        # DivIcon marker
        div_icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.DivIcon(html=f'<div style="font-size: 10pt;">({lat_marker:.4f}, {lon_marker:.4f})</div>')
        )
        div_icon_marker.add_to(m)

        # Icon marker
        icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.Icon(color='green')
        )
        icon_marker.add_to(m)

# Add a rectangle to the map
rectangle_points = [
    [llcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], llcrn_markers[1]],  # Close the rectangle
]

folium.Polygon(
    locations=rectangle_points,
    color='black',  # Line color of the rectangle
    fill=True,
    fill_color='none',  # Fill color of the rectangle
    fill_opacity=0.4,  # Opacity of the fill color
    weight=1.5,  # Width of the rectangle border
).add_to(m)

# Save the map to an HTML file
m.save(f'True_weekly.html')
m

"""Weekly predicted data"""

import folium
import numpy as np
import branca.colormap as cm

# Create a map centered on Ireland
m = folium.Map(location=[53.349805, -6.26031], zoom_start=7)

ireland_bound = [-10.3401, -6.022 ,51.828500000000005 , 55.422] #Irealand Extent
llcrn,urcrn = (-10.3401, 51.828500000000005),(-6.022, 55.422) # Ireland bounds

# Create latitude and longitude grids
num_cols = int((urcrn[0] - llcrn[0]) // 0.05)
num_rows = int((urcrn[1] - llcrn[1]) // 0.05)
lat_coords = np.linspace(llcrn[1], urcrn[1], num_rows + 1)
lon_coords = np.linspace(llcrn[0], urcrn[0], num_cols + 1)
lat, lon = np.meshgrid(lon_coords, lat_coords)


avg_pred = np.array(avg_pred).astype(np.float64)

# Initialize a map
# m = folium.Map(location=[53.0, -7.5], zoom_start=7, tiles='Stamen Terrain')

# Convert the numpy array to a GeoJSON-like structure
features = []
for i in range(num_rows):
    for j in range(num_cols):
        polygon = [
            [lat[i, j], lon[i, j]],
            [lat[i + 1, j], lon[i + 1, j]],
            [lat[i + 1, j + 1], lon[i + 1, j + 1]],
            [lat[i, j + 1], lon[i, j + 1]],
            [lat[i, j], lon[i, j]]
        ]
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            },
            'properties': {
                'value': avg_pred[i, j]  # Use the corresponding value from the 'true' array
            }
        }
        features.append(feature)

# Define a color scale for the 'true' values
vmin = np.min(avg_pred)
vmax = np.max(avg_pred)
colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'], vmin=vmin, vmax=vmax)

# Add the GeoJSON layer to the map and set fill color based on 'true' values
folium.GeoJson(
    {
        'type': 'FeatureCollection',
        'features': features
    },
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['value']),
        'color': 'none',
        'fillOpacity': 0.6
    }
).add_to(m)

# Create a color bar legend with values
legend_html = """
    <div style="
        position: fixed;
        bottom: 360px; left: 520px; width: 70px; height: 134px;
        background-color: white;
        z-index: 9999; font-size: 12px;
        border: 2px solid grey; padding: 2px;
    ">
"""

num_values = 5  # Number of values to display in the color bar legend
value_range = np.linspace(vmin, vmax, num_values)

for value in value_range:
    color = colormap(value)
    label = f'{int(value)}'  # Format the value as desired
    legend_html += f'<p><span style="background-color:{color};">&nbsp;&nbsp;&nbsp;&nbsp;</span> {label}</p>'

legend_html += '</div>'

# Add the color bar legend to the map
m.get_root().html.add_child(folium.Element(legend_html))

# Define the number of rows and columns in the grid for markers
num_rows_markers = 3
num_cols_markers = 3

# Define the latitude and longitude bounds for the grid for markers
llcrn_markers = [51.7665, -10.3401]
urcrn_markers = [55.422, -6.022]

# Calculate the step size for latitude and longitude for markers
lat_step_markers = (urcrn_markers[0] - llcrn_markers[0]) / num_rows_markers
lon_step_markers = (urcrn_markers[1] - llcrn_markers[1]) / num_cols_markers

# Add markers with both DivIcon and Icon inside the rectangle
for i in range(num_rows_markers + 1):
    for j in range(num_cols_markers + 1):
        lat_marker = llcrn_markers[0] + i * lat_step_markers
        lon_marker = llcrn_markers[1] + j * lon_step_markers

        # DivIcon marker
        div_icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.DivIcon(html=f'<div style="font-size: 10pt;">({lat_marker:.4f}, {lon_marker:.4f})</div>')
        )
        div_icon_marker.add_to(m)

        # Icon marker
        icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.Icon(color='green')
        )
        icon_marker.add_to(m)

# Add a rectangle to the map
rectangle_points = [
    [llcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], llcrn_markers[1]],  # Close the rectangle
]

folium.Polygon(
    locations=rectangle_points,
    color='black',  # Line color of the rectangle
    fill=True,
    fill_color='none',  # Fill color of the rectangle
    fill_opacity=0.4,  # Opacity of the fill color
    weight=1.5,  # Width of the rectangle border
).add_to(m)

# Save the map to an HTML file
m.save(f'pred_weekly.html')
m

"""Calcute montly data"""

# Initialize variables to store cumulative values
total_inp_month = np.zeros_like(X_val[0][:, :, 0])
total_true_month = np.zeros((71, 86))
total_pred_month = np.zeros((71, 86))

# Loop through the month
for month in range(0, 30):
    # Extract data for the current week
    inp_month = X_val[month][:, :, 0]
    true_month = Y_val[month].reshape((71, 86))
    pred_month = val_outputs[month].reshape((71, 86))
    #print(pred_month)
    # Accumulate values element-wise
    np.add(total_inp_month, inp_month, out=total_inp_month)
    np.add(total_true_month, true_month, out=total_true_month)
    np.add(total_pred_month, pred_month, out=total_pred_month)

# Calculate average values
avg_inp_month = total_inp_month / 30
avg_true_month = total_true_month / 30
avg_pred_month = total_pred_month / 30
#print(avg_pred_month)

"""Montly true data"""

import folium
import numpy as np
import branca.colormap as cm

# Create a map centered on Ireland
m = folium.Map(location=[53.349805, -6.26031], zoom_start=7)

ireland_bound = [-10.3401, -6.022 ,51.828500000000005 , 55.422] #Irealand Extent
llcrn,urcrn = (-10.3401, 51.828500000000005),(-6.022, 55.422) # Ireland bounds

# Create latitude and longitude grids
num_cols = int((urcrn[0] - llcrn[0]) // 0.05)
num_rows = int((urcrn[1] - llcrn[1]) // 0.05)
lat_coords = np.linspace(llcrn[1], urcrn[1], num_rows + 1)
lon_coords = np.linspace(llcrn[0], urcrn[0], num_cols + 1)
lat, lon = np.meshgrid(lon_coords, lat_coords)


avg_true_month = np.array(avg_true_month)

# Initialize a map
# m = folium.Map(location=[53.0, -7.5], zoom_start=7, tiles='Stamen Terrain')

# Convert the numpy array to a GeoJSON-like structure
features = []
for i in range(num_rows):
    for j in range(num_cols):
        polygon = [
            [lat[i, j], lon[i, j]],
            [lat[i + 1, j], lon[i + 1, j]],
            [lat[i + 1, j + 1], lon[i + 1, j + 1]],
            [lat[i, j + 1], lon[i, j + 1]],
            [lat[i, j], lon[i, j]]
        ]
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            },
            'properties': {
                'value': avg_true_month[i, j]  # Use the corresponding value from the 'true' array
            }
        }
        features.append(feature)

# Define a color scale for the 'true' values
vmin = np.min(avg_true_month)
vmax = np.max(avg_true_month)
colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'], vmin=vmin, vmax=vmax)

# Add the GeoJSON layer to the map and set fill color based on 'true' values
folium.GeoJson(
    {
        'type': 'FeatureCollection',
        'features': features
    },
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['value']),
        'color': 'none',
        'fillOpacity': 0.6
    }
).add_to(m)

# Create a color bar legend with values
legend_html = """
    <div style="
        position: fixed;
        bottom: 360px; left: 520px; width: 70px; height: 134px;
        background-color: white;
        z-index: 9999; font-size: 12px;
        border: 2px solid grey; padding: 2px;
    ">
"""

num_values = 5  # Number of values to display in the color bar legend
value_range = np.linspace(vmin, vmax, num_values)

for value in value_range:
    color = colormap(value)
    label = f'{int(value)}'  # Format the value as desired
    legend_html += f'<p><span style="background-color:{color};">&nbsp;&nbsp;&nbsp;&nbsp;</span> {label}</p>'

legend_html += '</div>'

# Add the color bar legend to the map
m.get_root().html.add_child(folium.Element(legend_html))

# Define the number of rows and columns in the grid for markers
num_rows_markers = 3
num_cols_markers = 3

# Define the latitude and longitude bounds for the grid for markers
llcrn_markers = [51.7665, -10.3401]
urcrn_markers = [55.422, -6.022]

# Calculate the step size for latitude and longitude for markers
lat_step_markers = (urcrn_markers[0] - llcrn_markers[0]) / num_rows_markers
lon_step_markers = (urcrn_markers[1] - llcrn_markers[1]) / num_cols_markers

# Add markers with both DivIcon and Icon inside the rectangle
for i in range(num_rows_markers + 1):
    for j in range(num_cols_markers + 1):
        lat_marker = llcrn_markers[0] + i * lat_step_markers
        lon_marker = llcrn_markers[1] + j * lon_step_markers

        # DivIcon marker
        div_icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.DivIcon(html=f'<div style="font-size: 10pt;">({lat_marker:.4f}, {lon_marker:.4f})</div>')
        )
        div_icon_marker.add_to(m)

        # Icon marker
        icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.Icon(color='green')
        )
        icon_marker.add_to(m)

# Add a rectangle to the map
rectangle_points = [
    [llcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], llcrn_markers[1]],  # Close the rectangle
]

folium.Polygon(
    locations=rectangle_points,
    color='black',  # Line color of the rectangle
    fill=True,
    fill_color='none',  # Fill color of the rectangle
    fill_opacity=0.4,  # Opacity of the fill color
    weight=1.5,  # Width of the rectangle border
).add_to(m)

# Save the map to an HTML file
m.save(f'True_monthly.html')
m

"""Monthly predicted data"""

import folium
import numpy as np
import branca.colormap as cm

# Create a map centered on Ireland
m = folium.Map(location=[53.349805, -6.26031], zoom_start=7)

ireland_bound = [-10.3401, -6.022 ,51.828500000000005 , 55.422] #Irealand Extent
llcrn,urcrn = (-10.3401, 51.828500000000005),(-6.022, 55.422) # Ireland bounds

# Create latitude and longitude grids
num_cols = int((urcrn[0] - llcrn[0]) // 0.05)
num_rows = int((urcrn[1] - llcrn[1]) // 0.05)
lat_coords = np.linspace(llcrn[1], urcrn[1], num_rows + 1)
lon_coords = np.linspace(llcrn[0], urcrn[0], num_cols + 1)
lat, lon = np.meshgrid(lon_coords, lat_coords)


avg_pred_month = np.array(avg_pred_month).astype(np.float64)

# Initialize a map
# m = folium.Map(location=[53.0, -7.5], zoom_start=7, tiles='Stamen Terrain')

# Convert the numpy array to a GeoJSON-like structure
features = []
for i in range(num_rows):
    for j in range(num_cols):
        polygon = [
            [lat[i, j], lon[i, j]],
            [lat[i + 1, j], lon[i + 1, j]],
            [lat[i + 1, j + 1], lon[i + 1, j + 1]],
            [lat[i, j + 1], lon[i, j + 1]],
            [lat[i, j], lon[i, j]]
        ]
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            },
            'properties': {
                'value': avg_pred_month[i, j]  # Use the corresponding value from the 'true' array
            }
        }
        features.append(feature)

# Define a color scale for the 'true' values
vmin = np.min(avg_pred_month)
vmax = np.max(avg_pred_month)
colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'], vmin=vmin, vmax=vmax)

# Add the GeoJSON layer to the map and set fill color based on 'true' values
folium.GeoJson(
    {
        'type': 'FeatureCollection',
        'features': features
    },
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['value']),
        'color': 'none',
        'fillOpacity': 0.6
    }
).add_to(m)

# Create a color bar legend with values
legend_html = """
    <div style="
        position: fixed;
        bottom: 360px; left: 520px; width: 70px; height: 134px;
        background-color: white;
        z-index: 9999; font-size: 12px;
        border: 2px solid grey; padding: 2px;
    ">
"""

num_values = 5  # Number of values to display in the color bar legend
value_range = np.linspace(vmin, vmax, num_values)

for value in value_range:
    color = colormap(value)
    label = f'{int(value)}'  # Format the value as desired
    legend_html += f'<p><span style="background-color:{color};">&nbsp;&nbsp;&nbsp;&nbsp;</span> {label}</p>'

legend_html += '</div>'

# Add the color bar legend to the map
m.get_root().html.add_child(folium.Element(legend_html))

# Define the number of rows and columns in the grid for markers
num_rows_markers = 3
num_cols_markers = 3

# Define the latitude and longitude bounds for the grid for markers
llcrn_markers = [51.7665, -10.3401]
urcrn_markers = [55.422, -6.022]

# Calculate the step size for latitude and longitude for markers
lat_step_markers = (urcrn_markers[0] - llcrn_markers[0]) / num_rows_markers
lon_step_markers = (urcrn_markers[1] - llcrn_markers[1]) / num_cols_markers

# Add markers with both DivIcon and Icon inside the rectangle
for i in range(num_rows_markers + 1):
    for j in range(num_cols_markers + 1):
        lat_marker = llcrn_markers[0] + i * lat_step_markers
        lon_marker = llcrn_markers[1] + j * lon_step_markers

        # DivIcon marker
        div_icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.DivIcon(html=f'<div style="font-size: 10pt;">({lat_marker:.4f}, {lon_marker:.4f})</div>')
        )
        div_icon_marker.add_to(m)

        # Icon marker
        icon_marker = folium.Marker(
            location=[lat_marker, lon_marker],
            icon=folium.Icon(color='green')
        )
        icon_marker.add_to(m)

# Add a rectangle to the map
rectangle_points = [
    [llcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], llcrn_markers[1]],
    [urcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], urcrn_markers[1]],
    [llcrn_markers[0], llcrn_markers[1]],  # Close the rectangle
]

folium.Polygon(
    locations=rectangle_points,
    color='black',  # Line color of the rectangle
    fill=True,
    fill_color='none',  # Fill color of the rectangle
    fill_opacity=0.4,  # Opacity of the fill color
    weight=1.5,  # Width of the rectangle border
).add_to(m)

# Save the map to an HTML file
m.save(f'pred_monthly.html')
m

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming pred and true are NumPy arrays with the correct shapes
# pred and true are (49, 67) in your case

# Flatten the arrays for plotting
flat_pred = pred.flatten()
flat_true = true.flatten()

# Create an array representing the concentration (you may need to adjust this based on your data)
concentration = np.linspace(0, 1, len(flat_pred))

# Create a scatter plot with a regression line and adjusted transparency and color
sns.regplot(x=flat_true, y=flat_pred, scatter_kws={'s': 40, 'alpha': 0.1,'color': 'blue'}, line_kws={'linestyle': '-.', 'color': 'black','linewidth': 1})
#sns.regplot(x=flat_true, y=flat_pred, scatter_kws={'c': ['blue', 'green', 'red']})
# Add labels and title
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Regression Plot: True vs. Predicted Values')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming pred and true are NumPy arrays with the correct shapes
# pred and true are (49, 67) in your case

# Flatten the arrays for plotting
flat_pred = pred.flatten()
flat_true = true.flatten()

# Create an array representing the concentration (you may need to adjust this based on your data)
concentration = np.linspace(0, 1, len(flat_pred))
#plt.figure(figsize=(6, 6))  # Adjust the width and height as needed

# Create a scatter plot with a regression line and adjusted transparency and color
#sns.regplot(x=flat_true, y=flat_pred, scatter_kws={'s': 40, 'alpha': 0.1,'color': 'blue'}, line_kws={'linestyle': '-.', 'color': 'black','linewidth': 1})
sns.jointplot(x=flat_true, y=flat_pred, kind='hex', gridsize=20)
#sns.regplot(x=flat_true, y=flat_pred, scatter_kws={'c': ['blue', 'green', 'red']})
# Add labels and title
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
#plt.title('Regression Plot: True vs. Predicted Values')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming pred and true are NumPy arrays with the correct shapes
# pred and true are (49, 67) in your case

# Flatten the arrays for plotting
flat_pred = avg_pred.flatten()
flat_true = avg_true.flatten()

# Create a scatter plot with a regression line and adjusted transparency
sns.regplot(x=flat_true, y=flat_pred, scatter_kws={'s': 40, 'alpha': 0.1,'color': 'blue'}, line_kws={'linestyle': '-.', 'color': 'black','linewidth': 1})

# Add labels and title
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Regression Plot: True vs. Predicted Values')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming pred and true are NumPy arrays with the correct shapes
# pred and true are (49, 67) in your case

# Flatten the arrays for plotting
flat_pred = avg_pred.flatten()
flat_true = avg_true.flatten()

# Create a scatter plot with a regression line and adjusted transparency
#sns.regplot(x=flat_true, y=flat_pred, scatter_kws={'s': 40, 'alpha': 0.1,'color': 'blue'}, line_kws={'linestyle': '-.', 'color': 'black','linewidth': 1})
sns.jointplot(x=flat_true, y=flat_pred, kind='hex', gridsize=20)
# Add labels and title
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
#plt.title('Regression Plot: True vs. Predicted Values')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming pred and true are NumPy arrays with the correct shapes
# pred and true are (49, 67) in your case

# Flatten the arrays for plotting
flat_pred = avg_pred_month.flatten()
flat_true = avg_true_month.flatten()

# Create a scatter plot with a regression line and adjusted transparency
sns.regplot(x=flat_true, y=flat_pred, scatter_kws={'s': 40, 'alpha': 0.1,'color': 'blue'}, line_kws={'linestyle': '-.', 'color': 'black','linewidth': 1})

# Add labels and title
plt.xlabel('True Values', fontsize=28)
plt.ylabel('Predicted Values', fontsize=28)
#plt.title('Regression Plot: True vs. Predicted Values')
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming pred and true are NumPy arrays with the correct shapes
# pred and true are (49, 67) in your case

# Flatten the arrays for plotting
flat_pred = avg_pred_month.flatten()
flat_true = avg_true_month.flatten()
print(flat_pred.shape)
print(flat_true.shape)

# Create a scatter plot with a regression line and adjusted transparency
#sns.regplot(x=flat_true, y=flat_pred, scatter_kws={'s': 40, 'alpha': 0.9,'color': 'blue'}, line_kws={'linestyle': '-.', 'color': 'black','linewidth': 1})
sns.jointplot(x=flat_true, y=flat_pred, kind='hex', gridsize=20)

# Add labels and title
plt.xlabel('True Values', fontsize=28)
plt.ylabel('Predicted Values', fontsize=28)
#plt.title('Regression Plot: True vs. Predicted Values')
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap



# Define custom colormap
colors = ['blue', 'green', 'yellow', 'orange', 'red']
cmap = ListedColormap(colors)

# Create a subplot for each day
fig, axs = plt.subplots(5, 13, figsize=(20, 8))

# Flatten the subplot array for easier iteration
axs = axs.flatten()

# Loop through each day and create a heatmap
for i in range(65):
    ax = axs[i]
    prediction = val_outputs[i].reshape((71, 86))
    im = ax.imshow(prediction, cmap=cmap)  # Use the custom colormap
    ax.set_title(f'Day {i + 1}',fontsize=16)
    ax.axis('off')

# Convert val_outputs to a flattened array
#val_outputs_flat = val_outputs.flatten()
val_outputs_flat = np.array(val_outputs).astype(np.float64)
# Calculate colorbar ticks based on the actual minimum and maximum values in the dataset
min_value = np.min(val_outputs_flat)
max_value = np.max(val_outputs_flat)
num_ticks = 5  # Show five ticks on the colorbar

# Add minimum and maximum values to the colorbar ticks
tick_positions = np.linspace(min_value, max_value, num_ticks)
tick_positions = np.insert(tick_positions, 0, min_value)  # Add min_value to the beginning
tick_positions = np.append(tick_positions, max_value)  # Add max_value to the end
tick_labels = [f'{int(value)}' for value in tick_positions]

# Add a colorbar for reference
cbar_ax = fig.add_axes([0.3, 0.01, 0.35, 0.015])  # Adjust these values as needed
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=tick_positions)
cbar.ax.tick_params(labelsize=16)
cbar.set_ticklabels(tick_labels)
#cbar.set_label('Concentration')  # Change this label based on your data

# Adjust layout for better spacing
plt.tight_layout()
plt.show()