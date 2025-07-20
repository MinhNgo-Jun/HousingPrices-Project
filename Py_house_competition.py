import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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

for col in df_test.columns:
    if df_test.isnull().sum()[col] > 0:
        print(f"Column '{col}' in test set has {df_test.isnull().sum()[col]} missing values.")

# Replace missing values categorical columns with 'None'
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna('None')

for col in df_test.select_dtypes(include=['object']).columns:
    df_test[col] = df_test[col].fillna('None')

# For numerical columns:
for col in df.columns:
    if df.isnull().sum()[col] > 0 and df[col].dtype in ['int64', 'float64']:
        print(f"Column '{col}' has {df.isnull().sum()[col]} missing values.")

for col in df_test.columns:
    if df_test.isnull().sum()[col] > 0 and df_test[col].dtype in ['int64', 'float64']:
        print(f"Column '{col}' in test set has {df_test.isnull().sum()[col]} missing values.")

# Replace missing values in 'LotFrontage' with the median
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].median())

# Replace missing values in 'MasVnrArea' with the zero value (0)
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(0)

# Replace missing values in 'GarageYrBlt' with the zero value (0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(0)

# Replace missing values in BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, BsmtFullBath, BsmtHalfBath, TotalBsmtSF, 
# GarageCars, GarageArea with zero value (0) in test datasets
for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', 'GarageCars', 'GarageArea']:
    df_test[col] = df_test[col].fillna(0)

# Check for missing values again
for col in df.columns:
    if df.isnull().sum()[col] > 0:
        print(f"Column '{col}' has {df.isnull().sum()[col]} missing values.")
    else:
        print(f"Column '{col}' has no missing values.")

for col in df_test.columns:
    if df_test.isnull().sum()[col] > 0:
        print(f"Column '{col}' in test set has {df_test.isnull().sum()[col]} missing values.")
    else:
        print(f"Column '{col}' in test set has no missing values.")

# Visualise the distribution of SalePrice
sns.histplot(df['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.show()

# Visualise the relationship between SalePrice and LotFrontage
sns.scatterplot(x='LotFrontage', y='SalePrice', data=df)
plt.title('SalePrice vs LotFrontage')
plt.show() 
## Outliers: LotFrontage higher than 300 or sale price higher than 700,000

# Visualise the relationship between SalePrice and LotArea
sns.scatterplot(x='LotArea', y='SalePrice', data=df)
plt.title('SalePrice vs LotArea')
plt.show()
## Outliers: LotArea higher than 150,000

# Visualise the relationship between SalePrice and living area
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.title('SalePrice vs GrLivArea')
plt.show()
## Outliers: GrLivArea higher than 4,000

# Visualise the relationship between SalePrice and MasVnrArea
sns.scatterplot(x='MasVnrArea', y='SalePrice', data=df)
plt.title('SalePrice vs MasVnrArea')
plt.show()
## Outliers: MasVnrArea higher than 1,300

# Visualise the relationship between SalePrice and GarageArea
sns.scatterplot(x='GarageArea', y='SalePrice', data=df)
plt.title('SalePrice vs GarageArea')
plt.show()
## Outliers: GarageArea higher than 1,200

# Remove outliers in SalePrice after visualisation
outliers = (df['LotFrontage'] > 300) | (df['SalePrice'] > 700000) | \
           (df['LotArea'] > 150000) | (df['GrLivArea'] > 4000) | \
           (df['MasVnrArea'] > 1300) | (df['GarageArea'] > 1200)
df = df[~outliers]

# Convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)
df_test_encoded = pd.get_dummies(df_test, drop_first=True)

# Split the dataset into training and testing sets
X = df_encoded.drop(['Id', 'SalePrice'], axis=1)
y = df_encoded['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Decision Tree Regressor Model
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train_scaled, y_train)

# Random Forest Regressor Model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_scaled, y_train)

# Boosted Decision Tree Model
boosted_tree_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
boosted_tree_model.fit(X_train_scaled, y_train)

# Support Vector Regressor Model
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)

# Neural Network Model
nn_model = MLPRegressor(hidden_layer_sizes=(128, 128, 64), max_iter=2000, random_state=42, 
                        learning_rate='constant', activation='relu',
                        alpha = 0.001, solver='adam')
nn_model.fit(X_train_scaled, y_train)

# Evaluate the models (using RMSE, R2 score and MAE)
models = {
    'Linear Regression': linear_model,
    'Decision Tree': decision_tree_model,
    'Random Forest': random_forest_model,
    'Boosted Decision Tree': boosted_tree_model,
    'Support Vector Regressor': svr_model,
    'Neural Network': nn_model
}

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)

## Fit the best model on the entire dataset
best_model = nn_model
best_model.fit(X_train_scaled, y_train)

# Predict on the test set
## Align the test set with the training set
df_test_encoded = df_test_encoded.reindex(columns=X_train.columns, fill_value=0)
X_test_final = scaler.transform(df_test_encoded)
y_test_pred = best_model.predict(X_test_final)

## Extract the predictions as csv file
sale_predict = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': y_test_pred})
sale_predict.to_csv('sale_predict.csv', index=False)