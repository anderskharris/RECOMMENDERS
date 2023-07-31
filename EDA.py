#%% 
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
from joblib import dump

#%%
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
df = session.sql('SELECT * FROM PURCHASE_DATA').to_pandas()

# %%
df.head()
# %%
