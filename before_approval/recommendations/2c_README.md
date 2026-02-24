## Paulo Bastos

## March 30th 2025

## Main Keypoints

> This document outlines the strategy for "recommending" new channels to existing hotels using exclusively past and neighboring behaviors (i.e., current hotel-channel combinations for each hotel/channel). As such, this versions does not take into account specific hotel/channel characteristics (e.g., size, price, geography).
>
> → `code` subfolder contains any code scripts, currently all under the `bpr_mf_channels_to_hotel.ipynb` notebook.
>
> → `data` subfolder contains any `input/other` data. This consists of a list of all observed hotel-channel combinations requests (ever).
>
> → `out` subfolder contains any `output` data. In it, the new recommendations for each hotel together with its score at available on the `bpr_mf_hotel_channel_recommendations_top50.csv` file.
>
> → `docs` subfolder contains any `documentation` for the present analysis. Start with the respective `README.html` file.

> → We implemented ** Personalized Ranking (BPR)** to suggest **new channels** for hotels. The goal was to identify channels that a hotel was **not yet interacting with**.

> Bayesian Personalized Ranking (BPR) is a popular recommendation system technique based on **matrix factorization**. It focuses on the **relative ranking** of items for a given user rather than absolute ratings. BPR is especially suitable for **implicit feedback data**, such as clicks, purchases, or views, where explicit ratings (e.g., 1-5 stars) are not available.

> ### Key Characteristics:
>
> - **Implicit feedback**: BPR works on implicit data (e.g., clicks, views, purchases), rather than explicit ratings.
> - **Pairwise ranking**: It optimizes the system to predict which items are more likely to be preferred by a user, rather than predicting an exact score.
> - **Matrix factorization**: BPR decomposes the user-item interaction matrix into lower-dimensional matrices to learn user and item latent factors.

> ## The BPR Model

> BPR is based on a **probabilistic** framework and aims to rank items such that more relevant items are ranked higher. The core idea is to model the relative preference between pairs of items, where a user is assumed to prefer one item over another.

> ### Assumptions:
>
> - Users (hotels) interact with some subset of items (e.g., channels).
> - Each hotel has "preferences" for channels that are not directly observable, but they can be inferred from past interactions.
> - For each hotel, the model learns two latent vectors: one for the hotel and one for the channel.

> ### Objective:
>
> For each hotel (user) `u` and for each **pair of items** `(i, j)`, the objective is to maximize the likelihood that the user prefers item `i` over item `j`, if the user has interacted with `i` but not `j`.

> ### Matrix Factorization:
>
> We assume that the user-item interaction matrix `R` is sparse, with only some interactions observed. BPR seeks to factorize this matrix into two low-rank matrices:

> - **User matrix** `U`: Where the rows are the users, and the columns are the latent factors.
> - **Item matrix** `V`: Where the rows are the items, and the columns are the latent factors.

> The latent factor vectors are learned by the algorithm for both users and items in the matrix factorization model.

> ## BPR Loss Function

> The BPR model uses a **pairwise ranking loss function**. Given a user `u`, a positively interacted item `i`, and a negatively interacted item `j`, the loss function aims to **maximize the difference** between the score for `i` and `j`:
>
> $$
> L_{BPR} = -\sum_{(u, i, j)} \log(\sigma(x_{ui} - x_{uj}))
> $$

> The goal is to **maximize** the log-likelihood that the user prefers item `i` over item `j`.

> ### Regularization:
>
> To prevent overfitting, **L2 regularization** is applied to the user and item latent factors to ensure that the model does not assign excessively large values to the latent factors. The regularization term is:

> $$
> L_{reg} = \lambda \left( \| U_u \|^2 + \| V_i \|^2 + \| V_j \|^2 \right)
> $$

> Thus, the final loss function to minimize is:

> $$
> L_{total} = L_{BPR} + L_{reg}
> $$

> ## Training the Model

> ### Optimization:
>
> The loss function is minimized using a **stochastic gradient descent (SGD)** approach. For each pair `(i, j)` of items for each user `u`, the gradients of the loss function with respect to the latent factors are computed and used to update the factors:

> $$
> U_u \leftarrow U_u - \eta \frac{\partial L_{total}}{\partial U_u}
> $$
>
> $$
> V_i \leftarrow V_i - \eta \frac{\partial L_{total}}{\partial V_i}
> $$
>
> $$
> V_j \leftarrow V_j - \eta \frac{\partial L_{total}}{\partial V_j}
> $$

> ### Iterations:
>
> The training continues for a pre-defined number of iterations (e.g., 5000 iterations), and the model converges when >the loss function stops improving.

> ## Inference (Making Recommendations)

> After training, to make recommendations for a user `u`, the model computes the scores for all items based on the learned latent factors:

> $$
> \hat{r}_{ui} = U_u^T V_i
> $$

> Then, the top-N items with the highest predicted scores are recommended to the user. These items have the highest likelihood of being preferred by the user.

## Python Script for the recommendation system

```python
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

import implicit
from implicit.bpr import BayesianPersonalizedRanking

```

```python
# Information about individual channels
data_lake_prd_314410_cz_canais = pd.read_csv('../data/lookups/data-lake-prd-314410.cz.canais.csv')

```

```python
# List of hotel-channel combinations as of January 2025
hotel_city_chanel_combin_extract  = pd.read_csv('../data/other/hotel_city_chanel_combin_extract.csv')
hotel_city_chanel_combin_extract.dropna(inplace=True)
hotel_city_chanel_combin_extract.drop(columns=['Cidade_ID'], inplace=True)
hotel_city_chanel_combin_extract.drop_duplicates(inplace=True)
```

> → Prepare the data as a squared matrix

```python
# Get unique hotel and channel IDs
hotel_ids = hotel_city_chanel_combin_extract["Hotel_ID"].unique()
channel_ids = hotel_city_chanel_combin_extract["Canal_ID"].unique()

# Create mappings
hotel_to_idx = {hotel_id: idx for idx, hotel_id in enumerate(hotel_ids)}
channel_to_idx = {channel_id: idx for idx, channel_id in enumerate(channel_ids)}

# Apply mappings
hotel_city_chanel_combin_extract["hotel_idx"] = hotel_city_chanel_combin_extract["Hotel_ID"].map(hotel_to_idx)
hotel_city_chanel_combin_extract["channels_idx"] = hotel_city_chanel_combin_extract["Canal_ID"].map(channel_to_idx)
```

```python
# Create a sparse interaction matrix
interaction_matrix = coo_matrix((
    np.ones(len(hotel_city_chanel_combin_extract)),
    (hotel_city_chanel_combin_extract["hotel_idx"], hotel_city_chanel_combin_extract["channels_idx"])
))


# Convert to CSR format for fast row access (required for BPR)
interaction_matrix_csr = interaction_matrix.tocsr()
```

> → Fit the model

```python
# Convert the matrix to a "positive feedback" format expected by implicit

interaction_matrix_csr = interaction_matrix_csr.astype("float32")

# Initialize the BPR model
model = BayesianPersonalizedRanking(factors=50, iterations=5000, regularization=0.01, learning_rate=0.001)

# Train the model on the interaction matrix
model.fit(interaction_matrix_csr, show_progress=True)
```

> → Get recommendations

```python
# Example: Get recommendations for a specific hotel (e.g., hotel ID 11334)

hotel_id = 11334
hotel_index = hotel_to_idx[hotel_id] # Convert to matrix index


# Reshape the matrix to a single row
user_interaction = interaction_matrix_csr[hotel_index]


# Generate top 10 channel recommendations
recommended_channel_indices, scores = model.recommend(hotel_index, user_interaction, N=50)
recommended_channel_indices
```

```python

# Convert back to original Canal_ID values
recommended_channels = [channel_ids[i] for i in recommended_channel_indices]

# Print results
print(f"Top recommended channels for Hotel {hotel_id}: {recommended_channels}")
print("")
print(f"Corresponding scores: {scores}")


```

```python

# Create a dictionary to store top recommendations for each hotel
top_recommendations = {}

# Loop through all hotels to get recommendations
for hotel_id in hotel_ids:
    hotel_index = hotel_to_idx[hotel_id] # Convert to matrix index

    # Reshape the matrix to a single row
    user_interaction = interaction_matrix_csr[hotel_index]

    # Generate top 50 channel recommendations
    recommended_channel_indices, scores = model.recommend(hotel_index, user_interaction, N=50)

    # Convert back to original Canal_ID values
    recommended_channels = [channel_ids[i] for i in recommended_channel_indices]

    # Get the channels that the hotel has already interacted with
    hotel_interacted_channels = hotel_city_chanel_combin_extract[hotel_city_chanel_combin_extract["Hotel_ID"] == hotel_id]["Canal_ID"].values

     # Exclude channels the hotel already interacted with
    filtered_recommended_channels = [
        channel for channel in recommended_channels if channel not in hotel_interacted_channels
    ]


    # Store the recommendations in the dictionary
    top_recommendations[hotel_id] = {
        'recommended_channels': filtered_recommended_channels,
        'scores': scores[:len(filtered_recommended_channels)]  # Adjust scores to match the filtered list
    }
```

```python
hotel_id_example = 2
if hotel_id_example in top_recommendations:
    print(f"Top recommended channels for Hotel {hotel_id_example}:")
    print(top_recommendations[hotel_id_example]['recommended_channels'])
    print("")
    print(f"Corresponding scores: {top_recommendations[hotel_id_example]['scores']}")
    print("")

else:
    print(f"No recommendations found for Hotel {hotel_id_example}")
```

> → Conver to table/csv

```python

# Prepare a list to store the recommendation data
recommendation_data = []

# Loop through the top_recommendations dictionary and flatten the data into rows
for hotel_id, recommendation in top_recommendations.items():
    for channel, score in zip(recommendation['recommended_channels'], recommendation['scores']):
        recommendation_data.append({
            'Hotel_ID': hotel_id,
            'Recommended_Channel': channel,
            'Score': score
        })

# Create a DataFrame from the recommendation data
recommendation_df = pd.DataFrame(recommendation_data)
```

```python

# Export the DataFrame to a CSV file
recommendation_df.to_csv('out/bpr_mf_hotel_channel_recommendations_top50.csv', index=False)

```

> # Notes
>
> Bayesian Personalized Ranking (BPR) recommendations are often quite different from those obtained by Singular Value Decomposition (SVD), and this difference can be traced back to their underlying principles and how they handle the data.

> **1. Objective Function (Optimization Focus)**
> BPR (Bayesian Personalized Ranking) is designed to optimize pairwise ranking of items for each user. It works on implicit feedback and aims to rank items relative to each other. Specifically, BPR tries to maximize the probability that a user prefers an item they have interacted with over one they have not. Mathematically, BPR is based on the logistic loss function that tries to differentiate the scores for items the user has interacted with and those they have not. The goal is relative ranking between items, not absolute ratings.

> Focus: BPR cares about the ordering of items (which items are better than others) for each user.

> Data type: Implicit feedback (e.g., binary interactions: clicked vs. not clicked).

> SVD (Singular Value Decomposition), on the other hand, is a matrix factorization technique that minimizes the reconstruction error between the original matrix and the product of latent factor matrices. SVD is generally used with explicit feedback (e.g., ratings on a scale of 1-5). SVD seeks to find a low-rank approximation of the user-item matrix that minimizes the difference between the actual ratings and the predicted ratings.

> Focus: SVD tries to predict the absolute rating of an item for a user, not just the ranking of items.

> Data type: Explicit feedback (e.g., ratings).

> **2. Handling of Implicit vs. Explicit Feedback**
> BPR is specifically tailored for implicit feedback. It is built to work with binary interactions (e.g., did a user interact with an item or not), where there is no direct rating information. BPR models the probability of user preferences based on whether they interacted with an item and treats non-interacted items as implicitly disliked.

> SVD is generally used for explicit feedback data, where each entry in the matrix represents a rating (e.g., 1 to 5 stars). SVD tries to predict the rating for each user-item pair by decomposing the matrix into latent factors and minimizing the difference between actual ratings and predicted ratings.

> Since BPR ignores the actual ratings and only cares about the relative preferences between items, it often produces a different set of recommendations than SVD, especially when explicit ratings data is available.

> **3. Latent Factor Representation**
> In BPR, latent factors for both users and items are learned based on relative preference. When a user interacts with item i, the model will try to make the latent factors of i more preferred compared to items the user hasn't interacted with (i.e., item j).

> The latent factors learned by **BPR** are personalized in terms of preference ranking, and BPR doesn’t necessarily predict a specific "rating" or score for an item.

> **This typically results in recommendations that emphasize diverse or unexpected items** because BPR is less concerned with predicting exact ratings and more focused on ordering preferences.

> In contrast, **SVD** learns latent factors by trying to reconstruct the original matrix. SVD aims to minimize the difference between predicted ratings and observed ratings, which means that SVD's latent factors are more focused on rating prediction. The **items recommended by SVD tend to be those with the highest predicted ratings.**

> Since SVD tries to approximate user ratings, it typically leads to more conservative recommendations, suggesting items that are similar to those the user has rated highly in the past.

> **4. Recommendation Differences**
> BPR will recommend items that might be more unexpected or novel, as the system tries to find items the user is likely to prefer relative to other items they have not interacted with.

> SVD will likely recommend items that are similar to the ones the user has previously rated highly, as the model is focused on predicting the exact rating for each item and tends to suggest items that fit within the user’s known preferences.

> **5. Bias Toward Popularity (SVD)** >**SVD may tend to recommend more popular items**, as those items tend to have the most data points and are therefore more easily predicted. If the user has rated items that are similar to popular ones, SVD will likely suggest those popular items.

> **BPR**, on the other hand, is not directly biased toward popularity, since it **cares more about individual user preferences and the ranking of items for each user**. As a result, BPR can sometimes recommend more niche or less popular items that the user may not have previously encountered.

> **Why the Difference in Recommendations?**

> BPR focuses on ranking items for a user based on previous interactions, which often leads to more exploratory recommendations (i.e., items the user might not have previously considered but would potentially like).
> SVD is focused on reconstructing ratings and may often suggest items that are similar to those the user has rated highly in the past. This can lead to more predictable or conservative recommendations.

> Exploration vs. Exploitation: In practice, systems that balance exploration (discovering new items via BPR) and exploitation (recommending top-rated items via SVD) are often more successful in providing a diverse yet relevant recommendation list.

> **Conclusion**

> The fundamental difference between BPR and SVD lies in their objective functions—BPR is focused on relative ranking based on implicit feedback, while SVD is focused on rating prediction based on explicit feedback. This results in BPR often suggesting novel and diverse items, while SVD tends to suggest popular and similar items based on a user’s past ratings. We suggest that, when in doubt, SVD suggestions should be prioritised.
