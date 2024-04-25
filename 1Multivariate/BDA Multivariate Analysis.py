

# ## **Importing Libraries**

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# ## **Read Data from CSV**

# In[3]:


dataset = pd.read_csv('/content/drive/MyDrive/BigDataAnalytics/Practical1_Multivariate/50_Startups.csv')
dataset


# ## **Preprocessing Data**

# In[4]:


dataset.isna().sum()


# In[5]:


dataset.info()


# In[6]:


X = dataset.drop('Profit', axis=1)
X


# In[7]:


y = dataset['Profit']
y


# ## **Encoding the categorical Features**

# In[8]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_feature = ["State"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  categorical_feature)],
                                 remainder="passthrough")

transformed_X = transformer.fit_transform(X)


# In[9]:


pd.DataFrame(transformed_X).head()


# ## **Performing Train Test Split**

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size = 0.25, random_state = 2509)


# ## **Applying Linear Regression on it**

# In[11]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[12]:


regressor.score(X_test,y_test)


# ## **Predicting the Outputs on Test data**

# In[13]:


y_pred = regressor.predict(X_test)


# In[14]:


d = {'y_pred': y_pred, 'y_test': y_test}


# In[15]:


pd.DataFrame(d)


# ## **Plotting the graph**
# 

# In[16]:


from google.colab import output
output.enable_custom_widget_manager()


# In[17]:


get_ipython().system('pip3 install ipympl')


# In[18]:


get_ipython().run_line_magic('matplotlib', 'widget')
# Convert DataFrame columns to NumPy arrays for plotting
x_rd_spend = X_test[:, 0]  # Assuming 'R&D Spend' is the first column in X_test
x_administration = X_test[:, 1]  # Assuming 'Administration' is the second column in X_test
y_actual = y_test
y_predicted = y_pred

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot actual data points
ax.scatter(x_rd_spend, x_administration, y_actual, color='r', label='Actual')

# Plot predicted data points
ax.scatter(x_rd_spend, x_administration, y_predicted, color='g', label='Predicted')

# Plot the regression plane
xx, yy = np.meshgrid(np.linspace(x_rd_spend.min(), x_rd_spend.max(), 10),
                     np.linspace(x_administration.min(), x_administration.max(), 10))
zz = regressor.coef_[0] * xx + regressor.coef_[1] * yy + regressor.intercept_

#ax.plot_surface(xx, yy, zz, alpha=0.5, color='b')

ax.plot_trisurf(xx.flatten(), yy.flatten(), zz.flatten(), alpha=0.5, color='b')

# Set labels and legend

ax.set_xlabel('R&D Spend')
ax.set_ylabel('Administration')
ax.set_zlabel('Profit')
ax.legend()

plt.show()


# Support for third party widgets will remain active for the duration of the session. To disable support:
