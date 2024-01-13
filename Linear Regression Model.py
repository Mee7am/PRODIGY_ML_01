#Importing some usefull libraries
import numpy
import pandas
import matplotlib.pyplot as plot
import seaborn as born
from sklearn.pipeline import make_pipeline as pipe
from sklearn.impute import SimpleImputer as imputer 
from sklearn.linear_model import LinearRegression as linear


train_path = "train.csv"
test_path = "test.csv"

#Loading training data to a frame
train_frame = pandas.read_csv(train_path)

#Selecting features for the frame
x_values = train_frame[['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'MSSubClass', 'MiscVal', 'MoSold', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold']]

#Creating pipeline with imputer and linear regression model
model = pipe(imputer(strategy = "mean"), linear())

#Training the model using the training features and target variables
model.fit(x_values, train_frame["SalePrice"])

#This loads the data frame
test_frame = pandas.read_csv(test_path)

#Selecting features for the frame
x_values = test_frame[['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'MSSubClass', 'MiscVal', 'MoSold', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold']]

#Making predictions on the test data
prediction = model.predict(x_values)

#Creating new data frame for submission
submission_df = pandas.DataFrame({
    "Id": test_frame["Id"],
    "SalePrice": prediction
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv("Predicted Values.csv", index=False)
