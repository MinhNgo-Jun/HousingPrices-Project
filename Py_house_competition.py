import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


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

# Outlier detection using IQR method and replace them by the boundaries
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    Q1 = df[col].quantile(0.25)
    Q2 = df[col].quantile(0.5)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

df_encoded.head()

# Split the dataset into training and testing sets
X = df_encoded.drop(['Id', 'SalePrice'], axis=1)
y = df_encoded['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
## Scale the numerical features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
linear_model.fit(X_train_scaled, y_train)
# Display the coefficients of the linear regression model
coefficients = pd.DataFrame(linear_model.coef_, X_train.columns, columns=['Coefficient'])
print(coefficients)
# Show top 10 high coefficients 
print(coefficients.nlargest(10, 'Coefficient'))
# Show coefficients starting with 'RoofMatl_'
print(coefficients[coefficients.index.str.startswith('RoofMatl_')])
### We can see that the RoofMatl has a significant impact on the SalePrice.

# Evaluate the Linear Regression model
y_pred_linear = linear_model.predict(X_test_scaled)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
print(f"RMSE for Linear Regression: {rmse_linear}")
### The RSME for Linear Regression is 27024.85114321822, which indicates a bad fit.

# Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the Random Forest model
y_pred_rf = rf_model.predict(X_test_scaled)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"RMSE for Random Forest: {rmse_rf}")
### The RSME for Random Forest is 19670.62687064665, which indicates a better fit than Linear Regression.

# Calculate ROC AUC score for Random Forest model
y_pred_rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_pred_rf_proba)
print(f"ROC AUC for Random Forest: {roc_auc_rf}")




# Neural Network Model 
## Neural networks includes 3 hidden layers with 256, 128, and 64 neurons respectively, .
nn_model = MLPRegressor(hidden_layer_sizes=(128, 128, 64), max_iter=500, random_state=42)
nn_model.fit(X_train_scaled, y_train)
