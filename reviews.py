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

# %% in snowpark
rvw = session.sql("select * from yelp_review_pa")
rsts = rvw.group_by("BUSINESS_ID").count().select("BUSINESS_ID").distinct().collect()
pvt_rvw = (
    rvw.select("USER_ID", "BUSINESS_ID", "STARS")
    .pivot("BUSINESS_ID", rsts)
    .avg("STARS")
)


# %% in python
rvw = session.sql(
    "select USER_ID, BUSINESS_ID, avg(STARS) as STARS from yelp_review_pa_train group by USER_ID, BUSINESS_ID"
).to_pandas()
pvt_rvw = rvw.pivot(index="USER_ID", columns="BUSINESS_ID", values="STARS")

# %%
