# %%
import yaml
import pandas as pd
import os
from snowflake.snowpark.session import Session
from snowflake.snowpark.types import IntegerType, Variant, VariantType, DecimalType
import snowflake.connector as snowflake
from snowflake.snowpark.functions import udf
from dotenv import load_dotenv
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from joblib import dump
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer

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

restaurants = session.sql("select * from yelp_business_pa").to_pandas()


# %%
# explore restaurant data
restaurants.describe()

# %% add map
fig = px.scatter_mapbox(
    restaurants,
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

# %% turning categories string into list
restaurants["CATEGORIES"] = (
    restaurants["CATEGORIES"].str.replace('"', "").str.replace(" ", "").str.split(",")
)

# %% looking at distribution of categories
categories = []
for i in restaurants["CATEGORIES"]:
    for r in i:
        categories.append(r)
categories = pd.DataFrame(categories, columns=["Category"])
categories = pd.DataFrame(
    categories.groupby(by=["Category"])["Category"]
    .count()
    .sort_values(ascending=False)
    .head(50)
)

# %%
# onehotencode categories
mlb = MultiLabelBinarizer(sparse_output=True, classes=list(categories.index))

restaurants = restaurants.join(
    pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(restaurants.pop("CATEGORIES")),
        index=restaurants.index,
        columns=mlb.classes_,
    )
)

# %%
