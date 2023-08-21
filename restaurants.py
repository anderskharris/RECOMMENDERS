# %%
import yaml
import pandas as pd
import os
import argparse
from snowflake.snowpark.session import Session
from snowflake.snowpark.types import IntegerType, Variant, VariantType, DecimalType
import snowflake.connector as snowflake
from snowflake.snowpark.functions import udf
from dotenv import load_dotenv
import numpy as np
from sklearn.cluster import DBSCAN
from joblib import dump
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer, PowerTransformer
import gower
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# %%
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open("secrets.yaml", "r") as f:
    secrets = yaml.load(f, Loader=yaml.FullLoader)

# %%
connection_parameters = {
    "account": config["account"],
    "user": secrets["username"],
    "password": secrets["password"],
    "role": config["role"],
    "warehouse": config["warehouse"],
    "database": config["database"],
    "schema": config["schema"],
}

session = Session.builder.configs(connection_parameters).create()
session.add_packages("snowflake-snowpark-python", "numpy", "scikit-learn", "pandas")

# %%
rst = session.sql("select * from yelp_business_pa").to_pandas()

# %% turning categories string into list
rst["CATEGORIES"] = (
    rst["CATEGORIES"].str.replace('"', "").str.replace(" ", "").str.split(",")
)

# %%
# explore restaurant data
rst.describe()

# %% add map
fig = px.scatter_mapbox(
    rst,
    lat="LATITUDE",
    lon="LONGITUDE",
    hover_name="NAME",
    hover_data=["STARS", "REVIEW_COUNT", "CITY"],
    color=str("CITY"),
    zoom=3,
    height=300,
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(mapbox_bounds={"west": -80, "east": -70, "south": 39, "north": 41})
fig.show()

# %%
# looking at distribution of categories and cities
categories = []
for i in rst["CATEGORIES"]:
    for r in i:
        categories.append(r)
categories = pd.DataFrame(categories, columns=["Category"])
categories = pd.DataFrame(
    categories.groupby(by=["Category"])["Category"]
    .count()
    .sort_values(ascending=False)
    .head(50)
)

cities = pd.DataFrame(
    rst.groupby(by=["CITY"])["CITY"].count().sort_values(ascending=False).head(75)
)

# %%
# onehotencode categories
mlb = MultiLabelBinarizer(sparse_output=False, classes=list(categories.index))

rst = rst.join(
    pd.DataFrame(
        mlb.fit_transform(rst.pop("CATEGORIES")).astype(bool),
        index=rst.index,
        columns=mlb.classes_,
    )
)

# %%
rst_train = rst.drop(columns=["BUSINESS_ID", "NAME", "LATITUDE", "LONGITUDE", "CITY"])
rst_train_num = rst_train.select_dtypes(exclude=["bool"])
rst_train_cat = rst_train.select_dtypes(include="bool")

cat_cols_bool = [
    col in rst_train.select_dtypes("bool").columns.to_list()
    for col in rst_train.columns.to_list()
]

# %% t-SNE Plotting


tsne = TSNE(n_components=2, random_state=0, perplexity=100)
projections = tsne.fit_transform(rst_train)

fig = px.scatter(projections, x=0, y=1, labels={"STARS": "REVIEW_COUNT"})
fig.show()


# %%
#####################
### UMAP PLOTTING ###
#####################

# Preprocessing numerical
numerical = rst_train.select_dtypes(exclude=["bool"])

for c in numerical.columns:
    pt = PowerTransformer()
    numerical.loc[:, c] = pt.fit_transform(np.array(numerical[c]).reshape(-1, 1))

# Preprocessing categorical
categorical = rst_train.select_dtypes(include="bool")
categorical = pd.get_dummies(categorical)

# Percentage of columns which are categorical is used as weight parameter in embeddings later
categorical_weight = (
    len(rst_train.select_dtypes(include="bool").columns) / rst_train.shape[1]
)

# Embedding numerical & categorical
fit1 = umap.UMAP(metric="l2").fit(numerical)
fit2 = umap.UMAP(metric="dice").fit(categorical)

# Augmenting the numerical embedding with categorical
intersection = umap.umap_.general_simplicial_set_intersection(
    fit1.graph_, fit2.graph_, weight=categorical_weight
)
intersection = umap.umap_.reset_local_connectivity(intersection)


plt.figure(figsize=(20, 10))
plt.scatter(*embedding.T, s=2, cmap="Spectral", alpha=1.0)
plt.show()


# %%
# training data and gower distance matrix
rst_train = rst.drop(columns=["BUSINESS_ID", "NAME", "LATITUDE", "LONGITUDE", "CITY"])

cat_cols_bool = [
    col in rst_train.select_dtypes("Sparse[int64, 0]").columns.to_list()
    for col in rst_train.columns.to_list()
]

dm = gower.gower_matrix(rst_train, cat_features=cat_cols_bool).astype(np.float64)
assert np.unique(np.diag(dm)) == 0

# %% DBSCAN

# Optimal Epsilon
print("Finding Optimal Epsilon...")
min_samples = 100
# SAMPLE_SIZE = 10000
# neigh = NearestNeighbors(n_neighbors=min_samples, metric="precomputed")
# nbrs = neigh.fit(dm)
# distances, indices = nbrs.kneighbors(dm)

#  Check shapes of Kneighbors matrices
# assert distances.shape == (SAMPLE_SIZE, min_samples)
# assert indices.shape == (SAMPLE_SIZE, min_samples)

# distances = np.sort(distances, axis=0)
# nearest_distances = distances[:, 1]

# knee = KneeLocator(
#     x=np.array(range(0, len(nearest_distances))),
#     y=nearest_distances,
#     curve="convex",
#     online=True,
# )
# EPS = knee.knee_y

# default EPS
EPS = 0.0005483931745402515

# Fit DBSCAN
print("Running DBSCAN...")
db = DBSCAN(eps=EPS, min_samples=min_samples, metric="precomputed").fit(dm)
print("DBSCAN FIT")

# %%
# need to visualize clusters to access quality.
