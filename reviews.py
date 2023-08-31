# %%
import yaml
import pandas as pd
import os
import argparse
from snowflake.snowpark.session import Session
import snowflake.connector as snowflake
from snowflake.snowpark.functions import udf
from dotenv import load_dotenv
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score


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

# %% data
rvw = session.sql(
    "select a.USER_ID, a.BUSINESS_ID, avg(a.STARS) - avg(b.AVG_STARS) as STARS from yelp_review_pa_train a LEFT JOIN (select USER_ID, avg(STARS) as AVG_STARS from yelp_review_pa_train group by USER_ID) b ON a.USER_ID = b.USER_ID group by a.USER_ID, a.BUSINESS_ID"
).to_pandas()
pvt_rvw = rvw.pivot(index="USER_ID", columns="BUSINESS_ID", values="STARS")

# %% SVD

SVD = TruncatedSVD(n_components=20, algorithm="arpack")
result_matrix = SVD.fit_transform(pvt_rvw)
result_matrix.shape

# %%
