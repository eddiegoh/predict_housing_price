# import the libraries needed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the data set
real_estate_evaluation_data = pd.read_excel('Real estate valuation data set.xlsx')

# Check if there is any missing data from any columns
real_estate_evaluation_data.info()

# Drop the serial number column 
real_estate_evaluation_data = real_estate_evaluation_data.drop(columns=['No'])

# Check that serial number column is dropped
real_estate_evaluation_data.info()

# Check for duplicated rows
real_estate_evaluation_data = real_estate_evaluation_data.drop_duplicates(subset=None, keep='first', inplace=False)
real_estate_evaluation_data.info()

# Look at the summary statistic for numeric columns to see if there are possible outlier
statistic = real_estate_evaluation_data.describe()

for i in range(7):
    print(statistic.iloc[:, i])

# Looking at the summary, there might be possible outliers in Y and X3
# But due to insufficient information we will not drop these rows
    
# Perform EDA to further analyse the data
sns.pairplot(real_estate_evaluation_data)
plt.show()

# Looking at pearsonr and p values to 
# examine the linear relationship and significant of correlation coefficient
# between independent variable against dependent variable
sns.jointplot(x='X3 distance to the nearest MRT station', y='Y house price of unit area',
              data=real_estate_evaluation_data)
plt.show()

sns.jointplot(x='X6 longitude', y='Y house price of unit area', data=real_estate_evaluation_data)
plt.show()

sns.jointplot(x='X5 latitude', y='Y house price of unit area', data=real_estate_evaluation_data)
plt.show()

sns.jointplot(x='X2 house age', y='Y house price of unit area', data=real_estate_evaluation_data)
plt.show()

sns.jointplot(x='X1 transaction date', y='Y house price of unit area', data=real_estate_evaluation_data)
plt.show()

sns.jointplot(x='X4 number of convenience stores', y='Y house price of unit area', data=real_estate_evaluation_data)
plt.show()

# Examine the correlation coefficient between variables 
correlation = stats.spearmanr(real_estate_evaluation_data, b=None, axis=0)
heat_map_matrix = correlation.correlation
sns.heatmap(heat_map_matrix, cmap='Blues', annot=True)
plt.show()

# looking at the correlation, linear relationship, significant of correlation coefficient 
# and the shape of the joint plot
# Since this is a house price forecasts problem, we will choose regression models
# we will pick X3, X4, X5 and X6 as our features to train the regression models
# we will do a comparison of the result with linear and polynomial regression model
# Split data in training and testing
X = real_estate_evaluation_data[['X3 distance to the nearest MRT station', 'X4 number of convenience stores',
                                 'X6 longitude', 'X5 latitude']]
y = real_estate_evaluation_data[['Y house price of unit area']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create linear regression objects
linear = linear_model.LinearRegression()
lin2 = linear_model.LinearRegression() 

# Cross-Validation to see if there is any considerable variation in our results
cv_results_linear = cross_val_score(linear, X, y, cv=3)
print('cv_results_linear:', cv_results_linear)
print(np.mean(cv_results_linear))
# Fitting Polynomial Regression to the data set
poly = PolynomialFeatures(degree = 2) 
X_train_poly = poly.fit_transform(X_train) 
X_test_poly = poly.fit_transform(X_test) 

# Cross-Validation to see if there is any considerable variation in our results
X_poly = poly.fit_transform(X) 
cv_results_poly = cross_val_score(lin2, X_poly, y, cv=3)
print('cv_results_poly:', cv_results_poly)
print(np.mean(cv_results_poly))
# Train the model using the training sets
linear.fit(X_train, y_train)

lin2.fit(X_train_poly, y_train) 

# Make predictions using the testing set
y_predict = linear.predict(X_test)
linear.score(X_test, y_test)
y_predict_poly = lin2.predict(X_test_poly)

# MSE was chosen as one of the metrics because of the present of outliers that we need to take into consideration
# Having MSE alone might not be sufficient to evaluate the model, R2 was chosen as it is closely related to MSE
# The mean squared error
print("Linear Regression Model Mean squared error: %.2f" % mean_squared_error(y_test, y_predict))
# Explained variance score: 1 is perfect prediction
print('Linear Regression Model Variance score: %.2f' % r2_score(y_test, y_predict))

# The mean squared error
print("Polynomial Regression Model Mean squared error: %.2f" % mean_squared_error(y_test, y_predict_poly))
# Explained variance score: 1 is perfect prediction
print('Polynomial Regression Model Variance score: %.2f' % r2_score(y_test, y_predict_poly))
print('Polynomial Regression was a better model as compared to linear Regression')