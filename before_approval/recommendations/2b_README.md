## Paulo Bastos

## March 30th 2025

## Main Keypoints

> This document outlines the strategy for "recommending" new channels to existing hotels using exclusively past and neighboring behaviors (i.e., current hotel-channel combinations for each hotel/channel). As such, this versions does not take into account specific hotel/channel characteristics (e.g., size, price, geography).
>
> → `code` subfolder contains any code scripts, currently all under the `autoencoder_channels_to_hotel.ipynb` notebook.
>
> → `data` subfolder contains any `input/other` data. This consists of a list of all observed hotel-channel combinations requests (ever).
>
> → `out` subfolder contains any `output` data. In it, the new recommendations for each hotel together with its score at available on the `autoencoder_hotel_channel_recommendations_top50.csv` file.
>
> → `docs` subfolder contains any `documentation` for the present analysis. Start with the respective `README.html` file.

> → We implemented **Autoencoders** to suggest **new channels** for hotels. The goal was to identify channels that a hotel was **not yet interacting with**.

> Autoencoders are neural networks that are trained to reconstruct their input data. They are especially powerful for tasks such as dimensionality reduction, anomaly detection, and recommendation systems.

> In the context of collaborative filtering, Autoencoders can:
>
> - Learn a **low-dimensional representation** of users and items (hotels and channels) in the hidden layers.
> - Reconstruct the interactions between users (hotels) and items (channels), allowing us to predict the interactions that have not been seen before.
> - Provide recommendations based on these reconstructions.
>
> Autoencoders help in capturing **non-linear patterns** in the data, which may not be possible with simpler linear models like SVD.

> ### How Autoencoders Work
>
> An Autoencoder is composed of two main components:
>
> 1.  **Encoder**: This part compresses the input data into a lower-dimensional space (also known as the bottleneck layer or latent space).
> 2.  **Decoder**: This part reconstructs the input data from the compressed representation produced by the encoder.
>
> The architecture can be visualized as follows:
>
> $$
> \text{Input Data} \xrightarrow{\text{Encoder}} \text{Latent Representation} \xrightarrow{\text{Decoder}} \text{Reconstructed Data}
> $$
>
> The goal of training an Autoencoder is to minimize the difference between the **input data** and the **reconstructed data**. This is achieved through **backpropagation** and an optimization process (e.g., Adam optimizer) using a loss function (e.g., Mean Squared Error).

> ### Mathematical Formulation
>
> The objective of the Autoencoder is to learn an encoding function that maps the input data to a lower-dimensional space, and a decoding function that maps the latent representation back to the original space.
>
> The Autoencoder is trained by minimizing the **reconstruction error**:
>
> $$
> \mathcal{L}(X, \hat{X}) = \frac{1}{m} \sum_{i=1}^{m} \| X_i - \hat{X}_i \|_2^2
> $$
>
> The loss function is optimized to minimize the difference between the original and reconstructed data, ensuring that the model captures the patterns in the data.

> > ### Model Architecture
>
> ### Encoder
>
> The **encoder** is a neural network that compresses the input data into a latent representation. In our case, the input data is a one-hot encoded vector representing the interactions of a hotel with all available channels. The encoder has two layers:
>
> - A fully connected layer that reduces the dimensionality from the input size (number of channels) to a smaller hidden dimension.
> - Another fully connected layer that reduces the representation further to half of the previous hidden size, providing a compact latent space.
>
> Mathematically, the encoder function is defined as:
>
> $$
> \mathbf{h} = f(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
> $$

> ### Decoder
>
> The **decoder** reconstructs the input data from the latent representation. It consists of two fully connected layers:
>
> - The first layer expands the latent space back to a larger dimension.
> - The second layer reconstructs the original data.
>
> Mathematically, the decoder function is defined as:
>
> $$
> \hat{\mathbf{x}} = g(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)
> $$

> ### Training
>
> The model is trained by feeding in the input interaction matrix (hotel-channel interaction data), which consists of binary values: 1 if a hotel has interacted with a channel, and 0 if not. The network tries to predict the missing values in the interaction matrix (i.e., recommending new channels).
>
> The training objective is to minimize the **Mean Squared Error** (MSE) between the original input and the reconstructed data. Once trained, the model can be used to predict the interaction matrix for hotels and generate recommendations based on the predicted values.

## Python Script for the recommendation system

```python
import pandas as pd
import numpy as np
import os


# Importing the data

# Information about individual channels
data_lake_prd_314410_cz_canais = pd.read_csv('../data/lookups/data-lake-prd-314410.cz.canais.csv')

# List of hotel-channel combinations as of January 2025
hotel_city_chanel_combin_extract  = pd.read_csv('../data/other/hotel_city_chanel_combin_extract.csv')
hotel_city_chanel_combin_extract.dropna(inplace=True)
hotel_city_chanel_combin_extract.drop(columns=['Cidade_ID'], inplace=True)
hotel_city_chanel_combin_extract.drop_duplicates(inplace=True)

```

> **NOTE**: The code below is for installing PyTorch with our specific GPU support
>
> Needs to be addapted on a different environment
>
> -Cuda compilation tools, release 12.5, V12.5.40
>
> -Build cuda_12.5.r12.5/compiler.34177558_0
>
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```python
import torch

print(torch.__version__)  # Check PyTorch version
print(torch.cuda.is_available())  # Should return True if GPU is detected
print(torch.cuda.get_device_name(0))  # Should print your GPU model

#2.5.1+cu121
#True
#NVIDIA RTX A2000 Laptop GPU
```

```python
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
```

> → Prepare the data as a squared matrix

```python
# Copy the dataset (long form list of hotels-channels combinations) to a new DataFrame
df = hotel_city_chanel_combin_extract.copy()

# Map Hotel_IDs and Channel_IDs to integer indices
hotel_ids = df["Hotel_ID"].unique()
channel_ids = df["Canal_ID"].unique()

hotel_to_idx = {hotel: i for i, hotel in enumerate(hotel_ids)}
channel_to_idx = {channel: i for i, channel in enumerate(channel_ids)}
```

```python
# Create interaction matrix (hotels × channels)
num_hotels = len(hotel_ids)
num_channels = len(channel_ids)
interaction_matrix = np.zeros((num_hotels, num_channels)) # hotel rows x channel columns
```

```python
# interaction_matrix
for _, row in df.iterrows():
    h_idx = hotel_to_idx[row["Hotel_ID"]]
    c_idx = channel_to_idx[row["Canal_ID"]]
    interaction_matrix[h_idx, c_idx] = 1
```

> → Split into train & test

```python
# Split into train & test
train_data, test_data = train_test_split(interaction_matrix, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_tensor = torch.FloatTensor(train_data)
test_tensor = torch.FloatTensor(test_data)

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(test_tensor), batch_size=64, shuffle=True)
```

> → Define the Autoencoder

```python
# Define the Autoencoder

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # Outputs between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

```

```python
# initialize the model
input_dim = num_channels # Each row is a hotel, so input_dim = number of channels ("columns")
autoencoder = Autoencoder(input_dim)
```

```python
autoencoder

'''
Autoencoder(
  (encoder): Sequential(
    (0): Linear(in_features=732, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ReLU()
  )
  (decoder): Sequential(
    (0): Linear(in_features=64, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=732, bias=True)
    (3): Sigmoid()
  )
)
'''
```

```python

# Loss function & optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
```

> → Train the Autoencoder

```python
# Train the Autoencoder
epochs = 200
for epoch in range(epochs):
    autoencoder.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        inputs = batch[0]  # Extract input tensor
        outputs = autoencoder(inputs)  # Forward pass
        loss = criterion(outputs, inputs)  # Compute loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
```

> → Generate recommendations

```python
# Generate recommendations
autoencoder.eval()
hotel_recommendations = {}

with torch.no_grad():
    reconstructed = autoencoder(torch.FloatTensor(interaction_matrix)).numpy()
```

```python
for hotel_idx, scores in enumerate(reconstructed):
    sorted_channels = np.argsort(scores)[::-1]  # Sort channels by predicted score (descending)
    recommended_channels = [channel_ids[i] for i in sorted_channels[:50]]  # Top 50 channels
    hotel_recommendations[hotel_ids[hotel_idx]] = recommended_channels
```

```python
# Show recommendations for a few hotels
for hotel, recs in list(hotel_recommendations.items())[:5]:
    print(f"Hotel {hotel} -> Recommended Channels: {recs}")
```

```python
Hotel 11332.0 -> Recommended Channels: [791, 248, 76, 315, 231, 1145, 889, 729, 481, 221, 390, 1351, 1199, 256, 1049, 976, 252, 136, 1195, 128, 385, 646, 1102, 1136, 1008, 970, 775, 937, 712, 901, 1020, 1055, 929, 122, 763, 269, 921, 463, 378, 641, 1360, 1306, 340, 292, 1254, 554, 367, 940, 170, 89]
Hotel 16573.0 -> Recommended Channels: [76, 124, 89, 270, 414, 763, 104, 223, 1248, 459, 1234, 972, 385, 699, 597, 942, 269, 1008, 1144, 494, 122, 689, 820, 1343, 1262, 1167, 865, 951, 1223, 162, 258, 684, 940, 126, 905, 369, 1046, 175, 987, 1138, 992, 964, 1055, 598, 1199, 921, 1141, 578, 418, 1034]
Hotel 12517.0 -> Recommended Channels: [1181, 672, 791, 1249, 646, 125, 231, 1234, 1108, 763, 89, 805, 1269, 595, 1210, 142, 1343, 175, 689, 277, 676, 87, 1200, 1099, 1360, 921, 369, 115, 986, 386, 122, 426, 30, 607, 157, 820, 1070, 269, 1093, 126, 1167, 1313, 1024, 475, 314, 684, 1165, 727, 783, 641]
Hotel 5882.0 -> Recommended Channels: [418, 1249, 396, 810, 1, 1055, 689, 627, 558, 994, 390, 964, 672, 1181, 974, 1273, 119, 385, 903, 125, 1008, 258, 684, 463, 838, 458, 527, 937, 122, 353, 987, 1349, 169, 172, 1138, 126, 763, 483, 905, 223, 791, 626, 1021, 554, 317, 921, 1360, 269, 969, 1248]
Hotel 5124.0 -> Recommended Channels: [136, 92, 1168, 1160, 1034, 964, 1056, 1024, 1099, 677, 1234, 1325, 1351, 358, 1063, 994, 1108, 1273, 10, 812, 914, 865, 1322, 1017, 175, 1199, 1197, 532, 1305, 659, 1144, 905, 71, 1346, 608, 369, 248, 641, 1248, 1008, 1360, 256, 131, 904, 805, 919, 597, 976, 1200, 284]
```

```python
hotel_recommendations.get(16573.0, [])
```

```python
hotel_recommendations
```

> → Convert dictionary to table/csv and exclude existing channels

```python
# Convert dictionary to DataFrame
df_recommendations = pd.DataFrame(
    [(hotel, channel) for hotel, channels in hotel_recommendations.items() for channel in channels],
    columns=['Hotel_ID', 'Channel_ID']
)

```

```python
df_recommendations["Hotel_ID"] = df_recommendations["Hotel_ID"].astype(int)
```

```python
existing_channels = hotel_city_chanel_combin_extract.groupby("Hotel_ID")["Canal_ID"].apply(set).to_dict()
```

```python
# Filter out already existing channels
filtered_recommendations = {
    hotel: [channel for channel in channels if channel not in existing_channels.get(hotel, set())]
    for hotel, channels in hotel_recommendations.items()
}
```

```python
# Convert to DataFrame
df_filtered_recommendations = pd.DataFrame([
    (hotel, channel) for hotel, channels in filtered_recommendations.items() for channel in channels
], columns=["Hotel_ID", "Recommended_Channel"])

df_filtered_recommendations
```

```python
df_filtered_recommendations.to_csv("../out/autoencoder_hotel_channel_recommendations_top50.csv", index=False)
```

> ## Sanity check
>
> Check if "similar" hotels are getting "similar" recommendations.
>
> It turns out that using Autoencoders this is not the case.
>
> **NOTE:** Autoencoders compress the hotel-channel interaction matrix into a lower-dimensional latent space (encoder), to reconstruct the interaction matrix from that latent representation (decoder) and to minimize reconstruction error.
>
> This means that Autoencoders learn to "rebuild" the original matrix rather than directly capturing high-variance structures (as SVD does).
>
> - In SVD, hotels with similar interaction patterns tend to get similar recommendations, because the singular vectors capture global patterns in the data.
> - In Autoencoders, the learned latent space might not be structured in the same way, so similar hotels might have very different activations in the bottleneck layer.
>
> As such, we prefer to exploy methologies for capturing high-variance structures.

```python
# Ensure model is in evaluation mode
autoencoder.eval()

# Pass the full dataset through the encoder to get latent features
with torch.no_grad():
    encoded_hotels = autoencoder.encoder(torch.FloatTensor(interaction_matrix)).numpy()  # Extract the encoded representation
```

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity between hotels based on their latent representations
similarity_matrix = cosine_similarity(encoded_hotels)

# Get the most similar hotels for a given hotel
hotel_id = 11332  # Example hotel

if hotel_id in hotel_to_idx:  # Ensure the hotel exists in our mapping
    similar_hotels = similarity_matrix[hotel_to_idx[hotel_id]].argsort()[-10:][::-1]
    print("Most similar hotels:", similar_hotels)
else:
    print(f"Hotel ID {hotel_id} not found in hotel_to_idx.")
```

```python

# Get the channel IDs for each hotel
channels_hotel_11332 = set(df_filtered_recommendations[df_filtered_recommendations["Hotel_ID"] == 11332]["Recommended_Channel"])
channels_hotel_13558 = set(df_filtered_recommendations[df_filtered_recommendations["Hotel_ID"] == 13558]["Recommended_Channel"])

# Compute the intersection of the two sets (overlap)
overlap_channels = channels_hotel_11332.intersection(channels_hotel_13558)

# Output the results
print(f"Channels for Hotel 11332: {channels_hotel_11332}")
print(f"Channels for Hotel 13558: {channels_hotel_13558}")
print(f"Overlap Channels: {overlap_channels}")
```

```python
Channels for Hotel 11332: {641, 901, 269, 921, 1306, 929, 292, 937, 554, 170, 940, 712, 970, 463, 1360, 340, 89, 122, 1254, 367, 1008, 1136, 248, 378, 763, 1020}
Channels for Hotel 13558: {256, 385, 258, 390, 908, 1165, 270, 783, 1167, 532, 792, 1181, 30, 1311, 32, 292, 937, 939, 1070, 432, 440, 314, 318, 575, 965, 87, 863, 608, 127, 1008, 122, 1020, 255}
Overlap Channels: {292, 937, 1008, 122, 1020}
```
