#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


data = pd.read_csv("C:/Users/SSC Destop/Desktop/BDA datasets/twitter/train_1600000.csv", encoding = "ISO-8859-1", engine="python")
data.columns = ["label", "time", "date", "query", "username", "text"]


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.columns


# In[ ]:




