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

# Display the number of missing values in each column
for col in df.columns:
    if df.isnull().sum()[col] > 0:
        print(f"Column '{col}' has {df.isnull().sum()[col]} missing values.")

# Replace missing values categorical columns with 'None'
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna('None')

# For numerical columns:
for col in df.columns:
    if df.isnull().sum()[col] > 0 and df[col].dtype in ['int64', 'float64']:
        print(f"Column '{col}' has {df.isnull().sum()[col]} missing values.")

# Replace missing values in 'LotFrontage' with the median
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# Replace missing values in 'MasVnrArea' with the zero value (0)
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Replace missing values in 'GarageYrBlt' with the zero value (0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

# Check for missing values again
for col in df.columns:
    if df.isnull().sum()[col] > 0:
        print(f"Column '{col}' has {df.isnull().sum()[col]} missing values.")
    else:
        print(f"Column '{col}' has no missing values.")

# Visualise the distribution of SalePrice
sns.histplot(df['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.show()

# Visualise the relationship between SalePrice and LotFrontage
sns.scatterplot(x='LotFrontage', y='SalePrice', data=df)
plt.title('SalePrice vs LotFrontage')
plt.show()

# Visualise the relationship between SalePrice and LotArea
sns.scatterplot(x='LotArea', y='SalePrice', data=df)
plt.title('SalePrice vs LotArea')
plt.show()

# Visualise the relationship between SalePrice and living area
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.title('SalePrice vs GrLivArea')
plt.show()

# Visualise the relationship between SalePrice and MasVnrArea
sns.scatterplot(x='MasVnrArea', y='SalePrice', data=df)
plt.title('SalePrice vs MasVnrArea')
plt.show()

# Visualise the relationship between SalePrice and GarageArea
sns.scatterplot(x='GarageArea', y='SalePrice', data=df)
plt.title('SalePrice vs GarageArea')
plt.show()

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Calculate the correlation matrix
correlation_matrix = df.corr()

