import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Load the dataset
df= pd.read_csv('Data/train.csv')
df_test = pd.read_csv("Data/test.csv")

# Display the first few rows and info of the dataset
df.head
df.info()

# Check for missing values
print(df.isnull().sum())

# Replace missing values in numerical columns with the median and in categorical columns with 'None'
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('None')
    elif df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())

print(df.isnull().sum())


