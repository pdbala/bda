
# ## **1.Importing Libraries**

# In[ ]:


#Lets import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# ## **2. Data Acquisition and Description**

# In[ ]:


#Let's load our train and test dataset
train = pd.read_csv('/content/drive/MyDrive/BigDataAnalytics/Practical5/train.csv')
test = pd.read_csv('/content/drive/MyDrive/BigDataAnalytics/Practical5/test.csv')
sample_submission = pd.read_csv('/content/drive/MyDrive/BigDataAnalytics/Practical5/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


#Lets look at the shape of your dataset
train.shape


# In[ ]:


# Lets look at the train.info()
train.info()


# In[ ]:


# Check for missing values
train.isna().sum().sort_values(ascending=False)


# In[ ]:


#Lets look at the discription also
train.describe()


# In[ ]:


# Lets look at the skewness of the numerical columns
num_columns = train.select_dtypes(include=['float64','int64'])
skewness = num_columns.apply(lambda x:x.skew())
print('Skewness of the numerical columns: \n',skewness)


# In[ ]:


#Lets check for duplicate rows of data
train.duplicated().any()


# In[ ]:


#Lets check for Cardinality
train.nunique()


# **Observations**:
# - Shape of our Dataset is : (8523, 12)
# - We have missing values in : Outlet_Size & Item_Weight .
# - **Item_Outlet_Sales** is our target column.
# - Few of the features have very less skewness. Data is almost normally distributed. No column with very high skewness.
# - We don't have any duplicate rows.
# - Few features have very high cardinality like **Item_Visibility,Item_MRP,Item_Outlet_Sales**.
# - dtype of all the columns looks fine.
# 

# ## **3.Exploratory Data Analysis**

# In[ ]:


# Before imputing the missing values let's do a simple EDA to check any inconsistency or outliers in our dataset


# In[ ]:


# Lets look at the columns Item_Visibility and Item_Fat_Content
sns.boxplot(x = train['Item_Visibility'], y = train['Item_Fat_Content'])
plt.show()


# In[ ]:


# Lets look at the column Item_Fat_Content
train['Item_Fat_Content'].value_counts().plot(kind='barh')
plt.show()


# In[ ]:


# Lets look at Item_Type column
train['Item_Type'].value_counts().plot(kind='barh')
plt.show()


# In[ ]:


# Lets look at the heatmap for the co-relation
sns.heatmap(num_columns.corr(),annot=True)
plt.show()


# **Observations**:
# - We have few inconsistencies in Item_Fat_Content
# - Few columns have some amount of co-relation like Item_MRP and Item_Outlet_Sales.

# ## **4.Data Preprocessing**

# In[ ]:


#Now lets prepare our Data for Modeling


# In[ ]:


# I have made a User defined function to impute the missing values
# and remove isconsistencies in 'Item_Fat_Content'
# Lets also make a new column to check the Years of Service


# In[ ]:


def data_prep(train):
  train['Item_Weight'] = np.where(train['Item_Weight'].isna(),train['Item_Weight'].median(skipna = True),train['Item_Weight'])
  train['Outlet_Size'] = np.where(train['Outlet_Size'].isna(),train['Outlet_Size'].mode()[0], train['Outlet_Size'])
  train['Item_Fat_Content'] = train['Item_Fat_Content'].replace('low fat', 'Low Fat')
  train['Item_Fat_Content'] = train['Item_Fat_Content'].replace('LF', 'Low Fat')
  train['Item_Fat_Content'] = train['Item_Fat_Content'].replace('reg', 'Regular')
  train['YOB'] = 2023 - train['Outlet_Establishment_Year']
  return train


# In[ ]:


train_new = data_prep(train)


# In[ ]:


train_new.info()


# In[ ]:


#Lets look at the 'Item_Fat_Content'
train['Item_Fat_Content'].value_counts()


# In[ ]:


# Lets do a chi-square test to check the co-relation


# In[ ]:


train['Item_Fat_Content'].value_counts()


# In[ ]:


train_new['Outlet_Size'].value_counts()


# In[ ]:


train_new['Outlet_Location_Type'].value_counts()


# In[ ]:


pd.crosstab(train_new['Outlet_Size'], train_new['Outlet_Location_Type'])


# In[ ]:


# Lets import chi2 contingency
from scipy.stats import chi2_contingency


# In[ ]:


# Lets define a function for chi square test
def chi_sq_test(var1, var2):
  cont_table = pd.crosstab(var1, var2)
  _,p,_,_ = chi2_contingency(cont_table)
  if p < 0.05:
    print('Accept the Alternate Hypothesis (There is a realation between var1 and var2)', round(p, 2))
  else:
    print('Failed to Reject Null Hypothesis (There is no relation between var1 and var2)', round(p, 2))
  return cont_table


# In[ ]:


chi_sq_test(train_new['Outlet_Size'], train_new['Outlet_Location_Type'])


# In[ ]:


chi_sq_test(train_new['Item_Fat_Content'], train_new['Item_Type'])


# ## **5.Data Preparation**

# In[ ]:


# Lets prepare our Data for Modelling


# In[ ]:


train.info()


# In[ ]:


train_new.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], inplace = True, axis = 1)


# In[ ]:


train_new.columns


# In[ ]:


train_new.info()


# In[ ]:


train_new = pd.get_dummies(train_new, columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size','Outlet_Location_Type','Outlet_Type'])


# In[ ]:


train_new.head()


# In[ ]:


train_new.info()


# ## **6.Data Modelling**

# In[ ]:


#First things first
# Lets devide our Dataset in X and y


# In[ ]:


y = train_new['Item_Outlet_Sales']
x = train_new.drop(['Item_Outlet_Sales'], axis = 1)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 15)


# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### **6.1 Linear Regression**

# In[ ]:


lr = LinearRegression()
lr.fit(x_train, y_train)
lr_train = lr.predict(x_train)
lr_test = lr.predict(x_test)


# In[ ]:


#Lets define a function for Model Evaluation
def model_eval(actual, predicted):
  rmse = np.sqrt(mean_squared_error(actual, predicted))
  r2 = r2_score(actual, predicted)
  print('The RMSE value for the model is: ', round(rmse,2))
  print('The R2 Score for the model is: ', round(r2, 2))


# In[ ]:


model_eval(y_train, lr_train)


# In[ ]:


model_eval(y_test, lr_test)


# ### **6.2 Random Forest Regressor**

# In[ ]:


rf = RandomForestRegressor()
rf.fit(x_train, y_train)


# In[ ]:


rf_preds_train = rf.predict(x_train)
rf_preds_test = rf.predict(x_test)


# In[ ]:


model_eval(y_train, rf_preds_train)


# In[ ]:


model_eval(y_test, rf_preds_test)


# ### **6.3 Ada Boost Regressor**

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor


# In[ ]:


ada = AdaBoostRegressor()
ada.fit(x_train, y_train)
preds_ada_train = ada.predict(x_train)
preds_ada_test = ada.predict(x_test)


# In[ ]:


model_eval(y_train, preds_ada_train)


# In[ ]:


model_eval(y_test, preds_ada_test)


# In[ ]:


# Lets look at the feature importance
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(x_train.columns, ada.feature_importances_):
    feats[feature] = importance


# In[ ]:


feats


# ### **6.4 Gradient Boosting Regressor**
# 
# 
# 

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


gb = GradientBoostingRegressor()
gb.fit(x_train, y_train)
preds_gb_train = gb.predict(x_train)
preds_gb_test = gb.predict(x_test)


# In[ ]:


model_eval(y_train, preds_gb_train)


# In[ ]:


model_eval(y_test,preds_gb_test )


# ### **6.5 XG Boost Regressor**

# In[ ]:


import xgboost as xg


# In[ ]:


xgb = xg.XGBRegressor()


# In[ ]:


xgb.fit(x_train, y_train)
preds_xgb_train = xgb.predict(x_train)
preds_xgb_test = xgb.predict(x_test)


# In[ ]:


model_eval(y_train, preds_xgb_train)


# In[ ]:


model_eval(y_test, preds_xgb_test)


# ### **6.5 SGD Regressor**

# In[ ]:


from sklearn.linear_model import SGDRegressor


# In[ ]:


sgd = SGDRegressor()
sgd.fit(x_train, y_train)
preds_train_sgd = sgd.predict(x_train)
preds_test_sgd = sgd.predict(x_test)


# In[ ]:


model_eval(y_train,preds_train_sgd )


# In[ ]:


model_eval(y_test, preds_test_sgd)


# ## **7.Hypertuning**

# In[ ]:


# Lets try and hypertune our model and see if we can further improve the RMSE and R2 Score


# In[ ]:


#Based on the bias variance trade off I have tried to hypertune Gradient Boosting Regressor using RandomizedSearchCV


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


# Number of estimators
n_estimators = [int(x) for x in np.linspace(start=50, stop=300, num=10)]

# Number of features to consider at every split
max_features = ['log2', 'sqrt']

# Maximum number of levels
max_depth = [int(x) for x in np.linspace(3, 15, num=5)] + [None]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Learning Rate
l_rate = [0.01, 0.05, 0.1, 0.5]


# In[ ]:


# Create the random grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'learning_rate': l_rate
}


# In[ ]:


# Randomized Search CV setup
gb_clf = RandomizedSearchCV(estimator=gb, param_distributions=random_grid, n_iter=100, cv=3, random_state=123, n_jobs=-1)


# In[ ]:


# Lets fit the Randomized Search CV to our data
gb_clf.fit(x_train, y_train)


# In[ ]:


# Get the best parameters and best score
print("Best Parameters:", gb_clf.best_params_)
print("Best Score:", gb_clf.best_score_)


# In[ ]:


gb2 = GradientBoostingRegressor(n_estimators=105,min_samples_split= 2,min_samples_leaf= 1,
                                max_features='sqrt',max_depth= 3,learning_rate= 0.1)
gb2.fit(x_train, y_train)
preds_gb2_train = gb2.predict(x_train)
preds_gb2_test = gb2.predict(x_test)


# In[ ]:


model_eval(y_train, preds_gb2_train)


# In[ ]:


model_eval(y_test,preds_gb2_test )


# ## **8.Test Data Application**

# In[ ]:


test.info()


# In[ ]:


data_prep(test)


# In[ ]:


test['Item_Fat_Content'].value_counts()


# In[ ]:


test.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], inplace = True, axis = 1)


# In[ ]:


test_new = pd.get_dummies(test, columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size','Outlet_Location_Type','Outlet_Type'])


# In[ ]:


test_new.info()


# In[ ]:


test_new['Item_Outlet_Sales'] = gb2.predict(test_new)


# In[ ]:


test_new.head()


# In[ ]:


test_new['Item_Outlet_Sales'] = abs(test_new['Item_Outlet_Sales'])


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission['Item_Outlet_Sales'] = test_new['Item_Outlet_Sales']


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.to_csv('final_output.csv', index = False)


# In[ ]:


import pickle


# In[ ]:


filename = 'finalized_model.sav'
pickle.dump(gb, open(filename, 'wb'))


# ## **9.Plotting Graphs**

# In[ ]:


# Importing necessary libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# List of model predictions
predictions = {
    "Linear Regression": lr_test,
    "Random Forest Regressor": rf_preds_test,
    "AdaBoost Regressor": preds_ada_test,
    "Gradient Boosting Regressor": preds_gb_test,
    "XGBoost Regressor": preds_xgb_test,
    "SGD Regressor": preds_test_sgd,
    "Gradient Boosting Regressor (Tuned)": preds_gb2_test
}

# Plotting predicted vs actual values for each model
plt.figure(figsize=(15, 10))
for model, preds in predictions.items():
    plt.scatter(y_test, preds, label=model)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values for Regression Models")
plt.legend()
plt.show()


# In[ ]:


# Plotting predicted vs actual values for each model
plt.figure(figsize=(15, 10))
for model, preds in predictions.items():
    plt.scatter(y_test, preds, label=model)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values for Regression Models")
plt.legend()

# Setting common range for both x and y axes
plt.xlim(min(y_test), max(y_test))
plt.ylim(min(y_test), max(y_test))

plt.show()


# In[ ]:


# Plotting predicted vs actual values for each model
plt.figure(figsize=(15, 10))
for model, preds in predictions.items():
    plt.scatter(preds, preds, label=model)  # Plotting predicted values
plt.scatter(y_test, y_test, marker='x', color='black', label='Actual Values')  # Plotting actual values separately
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values for Regression Models")
plt.legend()

# Setting common range for both x and y axes
plt.xlim(min(y_test), max(y_test))
plt.ylim(min(y_test), max(y_test))

plt.show()

