

# House Price Prediction




#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import FastMarkerCluster
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression\n",
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
     
      
            
            
# Importing the dataset
data = pd.read_csv('https://raw.githubusercontent.com/rashida048/Datasets/master/home_data.csv')
data.head()
      
              
# droping the unnecessary columns such as id, date, zipcode , lat and long
data.drop(['id','date'],axis=1,inplace=True)

     
           
# checking for null values/missing values
data.isnull().sum()
      
data.nunique()
  
  
# Data Preprocessing
    
# changing float to integer
data['bathrooms'] = data['bathrooms'].astype(int)
data['floors'] = data['floors'].astype(int)
# renaming the column yr_built to age and changing the values to age
data.rename(columns={'yr_built':'age'},inplace=True)
data['age'] = 2023 - data['age']
# changing the column yr_renovated to renovated and changing the values to 0 and 1
data.rename(columns={'yr_renovated':'renovated'},inplace=True)
data['renovated'] = data['renovated'].apply(lambda x: 0 if x == 0 else 1)
      
# using simple feature scaling
data['sqft_living'] = data['sqft_living']/data['sqft_living'].max()
data['sqft_living15'] = data['sqft_living15']/data['sqft_living15'].max()
data['sqft_lot'] = data['sqft_lot']/data['sqft_lot'].max()
data['sqft_above'] = data['sqft_above']/data['sqft_above'].max()
data['sqft_basement'] = data['sqft_basement']/data['sqft_basement'].max()
data['sqft_lot15'] = data['sqft_lot15']/data['sqft_lot15'].max()
     
# Exploratory Data Analysis
   
# Correlation Matrix to find the relationship between the variables
      
# using correlation statistical method to find the relation between the price and other features
data.corr()['price'].sort_values(ascending=False)
   
       
plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),annot=True)
plt.show()
# Visualizing the coorelation with price
      
data.corr()['price'][:-1].sort_values().plot(kind='bar')

# Visulaizing the data
     
# visualizing the relation between price and sqft_living, sqft_lot, sqft_above, sqft_basement, sqft_living15, sqft_lot15, age, renovated, bedrooms, bathrooms, floors, waterfront, view, condition, grade
fig, ax = plt.subplots(4,4,figsize=(20,20))
sns.scatterplot( x = data['sqft_living'], y = data['price'],ax=ax[0,0])
sns.scatterplot( x = data['sqft_lot'], y = data['price'],ax=ax[0,1])
sns.scatterplot( x = data['sqft_above'], y = data['price'],ax=ax[0,2])
sns.scatterplot( x = data['sqft_basement'], y = data['price'],ax=ax[0,3])
sns.scatterplot( x = data['sqft_living15'], y = data['price'],ax=ax[1,0])
sns.scatterplot( x = data['sqft_lot15'], y = data['price'],ax=ax[1,1])
sns.lineplot( x = data['age'], y = data['price'],ax=ax[1,2])
sns.boxplot( x = data['renovated'], y = data['price'],ax=ax[1,3])
sns.scatterplot( x = data['bedrooms'], y = data['price'],ax=ax[2,0])
sns.lineplot( x = data['bathrooms'], y = data['price'],ax=ax[2,1])
sns.barplot( x = data['floors'], y = data['price'],ax=ax[2,2])
sns.boxplot( x = data['waterfront'], y = data['price'],ax=ax[2,3])
sns.barplot( x = data['view'], y = data['price'],ax=ax[3,0])
sns.barplot( x = data['condition'], y = data['price'],ax=ax[3,1])
sns.lineplot( x = data['grade'], y = data['price'],ax=ax[3,2])
sns.lineplot( x = data['age'], y = data['renovated'],ax=ax[3,3])
plt.show()
     
# Plotting the location of the houses based on longitude and latitude on the map
      
# adding a new column price_range and categorizing the price into 4 categories
data['price_range'] = pd.cut(data['price'],bins=[0,321950,450000,645000,1295648],labels=['Low','Medium','High','Very High'])
      
map = folium.Map(location=[47.5480, -121.9836],zoom_start=8)
marker_cluster = FastMarkerCluster(data[['lat', 'long']].values.tolist()).add_to(map)
map
      
      
# Train/Test Split
    
data.drop(['price_range'],axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data.drop('price',axis=1),data['price'],test_size=0.3,random_state=10)
     
# Model Training
# Using pipeline to combine the transformers and estimators and fit the model
     
input = [('scale',StandardScaler()),('polynomial', PolynomialFeatures(degree=2)),('model',LinearRegression())]
pipe = Pipeline(input)
pipe
      
       
#training the model
pipe.fit(X_train,y_train)
pipe.score(X_test,y_test)
      
        
#testing the model
pipe_pred = pipe.predict(X_test)
r2_score(y_test,pipe_pred)
      
# Ridge Regression
      
Ridgemodel = Ridge(alpha = 0.001)
Ridgemodel
     
# training the model
Ridgemodel.fit(X_train,y_train)
Ridgemodel.score(X_test,y_test)
      
  
#testing the model
r_pred = Ridgemodel.predict(X_test)
r2_score(y_test,r_pred)
    
    
# Random Forest 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor
  
    
# training the model
regressor.fit(X_train,y_train)
regressor.score(X_test,y_test)
     
     
#testing the model
yhat = regressor.predict(X_test)
r2_score(y_test,yhat)
    
# Model Evalution
   
# Distribution plot from the models predictions and the actual values
      
# displot of the actual price and predicted price for all models
fig, ax = plt.subplots(1,3,figsize=(20,5))
sns.distplot(y_test,ax=ax[0])
sns.distplot(pipe_pred,ax=ax[0])
sns.distplot(y_test,ax=ax[1])
sns.distplot(r_pred,ax=ax[1])
sns.distplot(y_test,ax=ax[2])
sns.distplot(yhat,ax=ax[2])
# legends
ax[0].legend(['Actual Price','Predicted Price'])
ax[1].legend(['Actual Price','Predicted Price'])
ax[2].legend(['Actual Price','Predicted Price'])
# model name as title
ax[0].set_title('Linear Regression')
ax[1].set_title('Ridge Regression')
ax[2].set_title('Random Forest Regression')
plt.show()
     
# Error Evaluation
   
#plot the graph to compare mae, mse, rmse for all models
fig, ax = plt.subplots(1,3,figsize=(20,5))
sns.barplot(x=['Linear Regression','Ridge Regression','Random Forest'],y=[mean_absolute_error(y_test,pipe_pred),mean_absolute_error(y_test,r_pred),mean_absolute_error(y_test,yhat)],ax=ax[0])
sns.barplot(x=['Linear Regression','Ridge Regression','Random Forest'],y=[mean_squared_error(y_test,pipe_pred),mean_squared_error(y_test,r_pred),mean_squared_error(y_test,yhat)],ax=ax[1])
sns.barplot(x=['Linear Regression','Ridge Regression','Random Forest'],y=[np.sqrt(mean_squared_error(y_test,pipe_pred)),np.sqrt(mean_squared_error(y_test,r_pred)),np.sqrt(mean_squared_error(y_test,yhat))],ax=ax[2])
# label for the graph
ax[0].set_ylabel('Mean Absolute Error')
ax[1].set_ylabel('Mean Squared Error')
ax[2].set_ylabel('Root Mean Squared Error')
plt.show()
      
# Accuracy Evaluation
      
# plot accuracy of all models in the same graph
fig, ax = plt.subplots(figsize=(7,5))
sns.barplot(x=['Linear Regression','Ridge Regression','Random Forest Regression'],y=[metrics.r2_score(y_test,pipe_pred),metrics.r2_score(y_test,r_pred),metrics.r2_score(y_test,yhat)])
ax.set_title('Accuracy of all models'
plt.show()
   
# Predicting the price of a new house

#input the values
bedrooms = 3
bathrooms = 2
sqft_living = 2000
sqft_lot = 10000
floors = 2
waterfront = 0
view = 0
condition = 3
grade = 8
sqft_above = 2000
sqft_basement = 0
yr_built = 1990
yr_renovated = 0
zipcode = 98001
lat = 47.5480
long = -121.9836
sqft_living15 = 2000
sqft_lot15 = 10000
      
           
    
#predicting the price using random forest 
price = regressor.predict([[bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,lat,long,sqft_living15,sqft_lot15]])
print('The price of the house is $',price[0])
# The price of the house is $ 1078694.0533333335      
    
