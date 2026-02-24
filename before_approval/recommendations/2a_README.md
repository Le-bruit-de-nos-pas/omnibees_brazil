## Paulo Bastos

## March 30th 2025

## Main Keypoints

> This document outlines the strategy for "recommending" new channels to existing hotels using exclusively past and neighboring behaviors (i.e., current hotel-channel combinations for each hotel/channel). As such, this versions does not take into account specific hotel/channel characteristics (e.g., size, price, geography).
>
> → `code` subfolder contains any code scripts, currently all under the `svd_channels_to_hotel.ipynb` notebook.
>
> → `data` subfolder contains any `input/other` data. This consists of a list of all observed hotel-channel combinations requests (ever).
>
> → `out` subfolder contains any `output` data. In it, the new recommendations for each hotel together with its score at available on the `svd_hotel_channel_recommendations_df_top50.csv` file. The `svd_recommendations_dict.pkl` contains a dictionary with this same thata in python native format. Only the top 50 recommendations (excluding exisitng channels for that specific hotel) are included.
>
> → `docs` subfolder contains any `documentation` for the present analysis. Start with the respective `README.html` file.

> → We implemented a **Singular Value Decomposition (SVD)-based recommendation system** to suggest **new channels** for hotels. The goal was to identify channels that a hotel was **not yet interacting with**, using latent factor analysis to predict potential interactions.
>
> → SVD is a powerful matrix factorization technique that decomposes a given interaction matrix into **lower-dimensional latent factors** representing hotels and > channels.
>
> The purpose of using SVD is to decompose a large, sparse matrix (such as a user-item or hotel-channel matrix) into smaller, dense matrices that represent the latent factors underlying the data.
>
> For example, in a hotel-channel recommendation system:
>
> - The rows represent hotels (users).
> - The columns represent channels (items).
> - The entries represent whether a hotel uses a particular channel or not (binary, 0 or 1), or how much they use the channel (ratings).
>
> SVD allows us to reduce the dimensionality of this matrix, uncover hidden relationships (latent factors) between hotels and channels, and predict missing values (channels a hotel might be interested in). By doing so, SVD helps generate accurate recommendations based on user (hotel) preferences and item (channel) characteristics, even for unknown combinations.

> ### Since the interaction matrix is **sparse** (many missing values), SVD helps **approximate missing interactions**.

> ### Conceptual Explanation
>
> SVD works by decomposing the original matrix `R` (of shape `m x n` where `m` is the number of users (hotels) and `n` is the number of items (channels)) into three smaller matrices:
>
> $$
> R = U \Sigma V^T
> $$
>
> ### What does this decomposition represent?
>
> 1. **(User Features Matrix)**: Each row of this matrix represents a hotel (user) in the latent factor space, capturing the latent characteristics that describe each hotel’s preferences or behaviors.
>
> 2. **(Singular Values Matrix)**: These values capture the importance of each latent factor in the decomposition. They are sorted in descending order, and larger values indicate that the corresponding latent factors are more significant.
>
> 3. **(Item Features Matrix)**: Each column of this matrix represents a channel (item) in the latent factor space, capturing the latent characteristics that describe each channel’s attributes or features.

> ### How Does SVD Help in Recommendations?
>
> SVD allows us to make predictions for missing values in the matrix (e.g., predicting how much a hotel would be expected to be paired with a certain channel that they have never used before). Once the decomposition is done, the predicted rating for a hotel-channel combination can be calculated as:
>
> $$
> \hat{R}_{ij} = U_i \Sigma V_j^T
> $$

> By multiplying these components together, we obtain an estimate of the rating (or preference) that a hotel would give to a channel.

> #### The decomposition aims to approximate the original matrix by the product of 3 new matrices. The latent factors in capture the underlying structure and relationships between users and items (or hotels and channels), even if the original data matrix is sparse.

> ### Practical Aspects of SVD
>
> 1. **Dimensionality Reduction**: The number of latent factors is typically much smaller than the number of rows and columns in the original matrix. This reduction in dimensions helps in making the computations more efficient and reveals patterns in the data that might otherwise be difficult to detect.
>
> 2. **Regularization**: In practice, regularization is often applied to prevent overfitting, particularly when dealing with sparse matrices. Regularization terms are added to the loss function to penalize large values in the latent factors.
>
> 3. **Alternating Least Squares (ALS)**: SVD is typically computed using iterative methods like **Alternating Least Squares (ALS)**, where the latent factors are updated alternately for users and items to minimize the reconstruction error of the original matrix.

> ### Steps in Training an SVD Model
>
> 1. **Matrix Decomposition**: The matrix is decomposed into 3 new matrices.
> 2. **Model Fitting**: The SVD model learns the latent factors that best represent the preferences or interactions between users and items.
> 3. **Prediction**: For each hotel (user) and channel (item), the model predicts the rating (or preference) using the learned latent factors.

## Python Script for the recommendation system

```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

from collections import defaultdict
import os
pd.set_option('display.float_format', lambda x: '%.0f' % x)

```

```python
# Information about individual channels
data_lake_prd_314410_cz_canais = pd.read_csv('../data/lookups/data-lake-prd-314410.cz.canais.csv')

# List of hotel-channel combinations as of January 2025
hotel_city_chanel_combin_extract  = pd.read_csv('../data/other/hotel_city_chanel_combin_extract.csv')
hotel_city_chanel_combin_extract.dropna(inplace=True)
hotel_city_chanel_combin_extract.drop(columns=['Cidade_ID'], inplace=True)
hotel_city_chanel_combin_extract.drop_duplicates(inplace=True)
```

```python
# Pivot the table
pivot_table = hotel_city_chanel_combin_extract.pivot_table(index='Hotel_ID', columns='Canal_ID', aggfunc='size', fill_value=0)
# Convert the table to binary (1 where the combination existed, 0 otherwise)
pivot_table = pivot_table.map(lambda x: 1 if x > 0 else 0)
```

```python
# Count the number of 1s for each column
counts_per_channel = pivot_table.sum().sort_values(ascending=False)
```

> → Prepare the data as a squared matrix

```python
# Step 1: Prepare the data matrix

# Loops through the rows (hotels) and columns (channels) of the wide matrix above
# Extracts ratings from the DataFrame and stores them as (hotel, channel, rating) tuples
# Creates a new Pandas DataFrame (ratings_df) with three columns:
# -hotel: The identifier of the hotel.
# -channel: The distribution channel (e.g., Expedia, Booking.com, etc.).
# -rating: The rating or score between 0 and 1.
# Prepares the data for the Surprise library. Reader(rating_scale=(0, 1)) tells Surprise that ratings range from 0 to 1.
# Dataset.load_from_df(ratings_df, reader) converts the DataFrame into a Surprise Dataset.


def prepare_data(df):
    ratings = []
    for hotel in df.index:
        for channel in df.columns:
            ratings.append((hotel, channel, df.loc[hotel, channel]))

    ratings_df = pd.DataFrame(ratings, columns=['hotel', 'channel', 'rating'])
    reader = Reader(rating_scale=(0, 1))
    return Dataset.load_from_df(ratings_df, reader)
```

> → Train the model

```python
# Step 2: Train the model

# Splits the Data into Training & Test Sets
# model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
# SVD is a matrix factorization model used in collaborative filtering.
#Parameters:
# n_factors=100 → Number of latent factors (hidden features) in the model.
# n_epochs=20 → Number of training iterations.
# lr_all=0.005 → Learning rate for gradient descent.
# reg_all=0.02 → Regularization term to prevent overfitting.


def train_model(data):
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    return model
```

> → Generate recommendations

```python
# Step 3: Generate recommendations

# Processes the prediction results from the SVD model and extracts the top N recommendations for each hotel.
# Creates a Dictionary to Store Recommendations
# top_n is a dictionary where:
# -Keys = uid (hotel ID).
# -Values = A list of tuples (iid, est), where:
# -iid = channel ID.
# -est = predicted rating.
# Processes Predictions and Stores Estimated Ratings
# for uid, iid, true_r, est, _ in predictions:
#    top_n[uid].append((iid, est))
# The predictions list contains tuples with:
# -uid: Hotel ID
# -iid: Channel ID
# -true_r: Actual rating (ignored in this function)
# -est: Predicted rating (used for ranking)
# Stores (iid, est) in top_n[uid] for each hotel.
# Sorts Channels by Predicted Rating in Descending Order
#  for uid, user_ratings in top_n.items():
#     user_ratings.sort(key=lambda x: x[1], reverse=True)
#      top_n[uid] = user_ratings[:n]
# Sorts the channels for each hotel based on estimated rating (est).
# Keeps only the top N channels with the highest predicted ratings.
# Returns the Dictionary with Top N Recommendations

def get_top_n_recommendations(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
```

> → Get recommendations

```python
# Step 4: Recommend channels

# Generates top N recommended channels for a given hotel using a trained SVD model.
# Extracts All Available Channels from the Dataset
# Creates a Test Set for the Given Hotel
# Generates a list of test samples where:
# -hotel_id: The hotel we want recommendations for.
# -iid: Each possible channel.
# -0: A placeholder rating (it will be predicted).
# Uses the SVD model to predict ratings for all channels.
# Returns the Top N Channels for the Hotel

def recommend_channels(hotel_id, model, data, n=5):
    iids = data.df['channel'].unique()
    testset = [(hotel_id, iid, 0) for iid in iids]
    predictions = model.test(testset)
    top_n = get_top_n_recommendations(predictions, n)
    return top_n[hotel_id]
```

```python
# Main execution

df = pivot_table
data = prepare_data(df)
model = train_model(data)
```

```python
# Example usage for 1 hotel
hotel_id = df.index[0]  # Choose a hotel

recommendations = recommend_channels(hotel_id, model, data, n=50)

print(f"Recommended channels for hotel {hotel_id:.0f}:")

for channel, score in recommendations:
    print(f"{channel}: {score:.4f}")
```

> → Note: Remove recommendated but already existing channels

```python

def recommend_channels_exclude_existing(hotel_id, model, data, existing_channels, n=50):
    # Get unique channel IDs from the data
    iids = data.df['channel'].unique()

    # Generate test set for the given hotel
    testset = [(hotel_id, iid, 0) for iid in iids]

    # Get predictions for the test set
    predictions = model.test(testset)

    # Get top N recommendations
    top_n = get_top_n_recommendations(predictions, n)

    # Get the list of channels that the hotel already has
    existing_hotel_channels = existing_channels[existing_channels['Hotel_ID'] == hotel_id]['Canal_ID'].values

    # Exclude the channels that are already associated with the hotel
    filtered_recommendations = [rec for rec in top_n[hotel_id] if rec[0] not in existing_hotel_channels]

    return filtered_recommendations
```

```python
# Create a dictionary to store the recommendations for each hotel
recommendations_dict = {}

# Loop through each hotel in df and get the top 50 recommended channels
for hotel_id in df.index:
    recommendations = recommend_channels_exclude_existing(hotel_id, model, data, hotel_city_chanel_combin_extract, n=50)

    # Store the recommendations in the dictionary
    recommendations_dict[hotel_id] = recommendations
```

```python
import pickle

# Save the recommendations_dict using pickle
# Pickle serializes (converts) Python objects into a binary format for storage or transfer
# Then deserializes (restores) them back to their original form when needed.
# Serialization (Pickling): The process of converting a Python object into a byte stream (binary data) that can be saved to a file or sent over a network.
# Deserialization (Unpickling): The process of reading a byte stream (binary data) and converting it back into a Python object.
# Pickle uses a binary format to represent Python objects (not human-readable).

with open('../out/recommendations_dict.pkl', 'wb') as f:
    pickle.dump(recommendations_dict, f)
```

```python
# Load the recommendations_dict using pickle
with open('../out/svd_recommendations_dict.pkl', 'rb') as f:
    loaded_recommendations_dict = pickle.load(f)
```

> → Convert the dictionary as a table/csv

```python
flattened_data = []

for hotel_id, recommendations in recommendations_dict.items():
    for channel_ID, score in recommendations:
        flattened_data.append({
            'Hotel_ID': hotel_id,
            'Channel_ID': channel_ID,
            'Score': score
        })


flattened_data = pd.DataFrame(flattened_data)
```

```python
pd.set_option('display.float_format', lambda x: '%.4f' % x)

flattened_data
```

```python
flattened_data['Hotel_ID'] = flattened_data['Hotel_ID'].astype(int)
flattened_data.to_csv('../out/svd_hotel_channel_recommendations_df_top50.csv', index=False)
```
