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

from xgboost import XGBRegressor
from tqdm import tqdm  # Import tqdm for the progress bar

scores_list = []
rmses = []
maes = []
pearsons = []

for fold in tqdm(range(5), desc="Folds Progress"):
  print("\nFold {}\n".format(fold))
  train_files = files[files["Fold"]!=fold]
  val_files = files[files["Fold"]==fold]

  X_train,Y_train = loadData(train_files)
  X_val,Y_val = loadData(val_files)
  #print("X_val=", X_val.shape)
  #print("Y_val=", Y_val.shape)

  # loss_plt = utils.loss_plt()
  model = XGBRegressor(objective="reg:squarederror")
  model.fit(X_train.reshape(-1,1),Y_train.reshape(-1,1))
  pred = model.predict(X_val.reshape(-1,1))
  #print("prediction=", pred.shape)
  Y_val = Y_val.reshape(Y_val.shape[0], -1)  # Reshape Y_val to (69, 3283)
  prediction = pred.reshape(Y_val.shape)
  #print("Prediction=", prediction.shape)
  rmse = mean_squared_error(Y_val,prediction,squared=False)
  mae = mean_absolute_error(Y_val,prediction)
  rmses.append(rmse)
  maes.append(mae)

  print("Fold {} RMSE Score: {}".format(fold, rmse))
  print("Fold {} MAE Score: {}".format(fold, mae))
  s, ls = pearsonCorrAvgDays(Y_val, prediction)
  print("Fold {} Pearson coeff avg over days: {}".format(fold, np.mean([i for i in ls if not pd.isnull(i)])))
  scores_list.append(ls)


print("\nCV RMSE Score: {}".format(np.mean(rmses)))
print("\nCV MAE Score: {}".format(np.mean(maes)))

print(Y_val.shape)
inp = X_val[10][:,:,0]
pred = prediction[10].reshape((71,86))
true = Y_val[10].reshape((71,86))

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
plt.xlabel('True Values', fontsize=28)
plt.ylabel('Predicted Values', fontsize=28)
#plt.title('Regression Plot: True vs. Predicted Values')
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

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
plt.xlabel('True Values', fontsize=28)
plt.ylabel('Predicted Values', fontsize=28)
#plt.title('Regression Plot: True vs. Predicted Values')
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Show the plot
plt.show()