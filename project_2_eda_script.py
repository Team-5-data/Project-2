#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-size:5em; font-family:fantasy; color:#00204a; font-weight:700;">Hotel Booking Project</h1>

# <h1 style="font-size:3em;  font-family:roboto, sans-serif; color:#005792; font-weight:700">Objective</h1>
# <ul style="font-size:1em; font-weight:400; color:#00bbf0">
#     <li>Import Libraries</li>
#     <li>Import Dataset</li>
#     <li>Understanding Dataset</li>
#     <li>Data Preperation</li>
#     <li>Feature transformation</li>
#     <li>New Feature Development</li>
#     <li>Exporting Clean Dataset</li>
# </ul>

# <h2 style="color:#fdb44b; font-size:2em">Libraries</h2>

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from datetime import date
from sklearn.preprocessing import LabelEncoder


# <h2 style="font-size:2em; color:#fdb44b;">Data Understanding and Preperation</h2> 

# In[68]:


# Fetching dataset 
df = pd.read_csv("hotel_bookings.csv")


# In[69]:


# data overview
pd.set_option("display.max_columns" , None) 
df.head()


# In[70]:


# Understanding dataset 
df.info()


# In[71]:


# required pattern's from dataset we have total count , average , standard of deviation value , min value , max value, q1 , q2 and q3
df.describe()


# In[72]:


# columns with null values
df.isna().sum()


# In[73]:


# Duplicated value
df.duplicated().sum()


# In[74]:


# removing 31994 rows
df.drop_duplicates(inplace=True)


# In[75]:


# value count of columns 
encoder={}
for i in df.columns:
    if df[i].dtype ==object:
        encoder[i]=df[i].value_counts().keys()
# need to fix reservation_status_date


# In[76]:


# checking null value again
df.isna().sum()


# In[77]:


# Filling null values
# Replace missing values safely
df["agent"]   = df["agent"].fillna(0)
df["company"] = df["company"].fillna(0)
df["children"] = df["children"].fillna(df["children"].mean())
df["country"]  = df["country"].fillna(df["country"].mode()[0])


# <h3 style="color:#edb1f1">Outlier Detention and Removal</h3>

# In[78]:


# Outlier Detention
for i in df.columns:
    if df[i].dtype !=object:
        sns.boxplot(df[i])
        plt.xlabel(i)
        plt.show()
# we will only focuses on removing outlier in average daily rate column


# In[79]:


# Outlier Removal
for i in outlier_col:
    # Quantile 1 and Quantile 3
    Q1= df["adr"].quantile(0.25)
    Q3= df["adr"].quantile(0.75)

    # Interquantile Range
    IQR = Q3-Q1

    # Upper and Lower fench value
    Upper = Q3 + 1.5 * IQR
    Lower = Q1 - 1.5 * IQR
    
    # filtering and putting back outlier removed value
    df = df[(df["adr"]>=Lower) & (df["adr"]<=Upper)]
    


# In[80]:


# checking outlier removed col
sns.boxplot(df["adr"])
plt.xlabel("Average Daily Rate")
plt.show()


# In[82]:


# fixing reservation_status_date
day=[]
month=[]
year=[]
rev_date = pd.to_datetime(df["reservation_status_date"])
for i in rev_date:
    # fetching day
    day.append(i.day)
    # fetching month
    month.append(i.month)
    # fetching year
    year.append(i.year)


# In[83]:


# putting back value 
df["reservation_status_day"]= day
df["reservation_status_month"]=month
df["reservation_status_year"]=year


# In[84]:


# dropping reservation_status_date
df.drop('reservation_status_date',axis=1,inplace=True)


# <h2 style="font-size:2em;color:#fdb44b;">Feature Engineering</h2>

# <h3 style="color:#edb1f1">Feature Transformation</h3>

# In[85]:


df_copy = df.copy() # copying dataset
encoder={}
for i in df_copy.columns:
    if df_copy[i].dtype == object:
        label = LabelEncoder()
        df_copy[i]=label.fit_transform(df_copy[i])
        encoder[i]=label


# <h3 style="color:#edb1f1;">Feature Development</h3>

# In[86]:


# Guest Total stays in hotel 
df_copy["guest_total_stay"]= df_copy["stays_in_weekend_nights"]+df_copy["stays_in_week_nights"]
# Long stay guest 
df_copy["is_stay_long"]=[ 1 if i > 7  else 0 for i in df_copy["guest_total_stay"] ]
# total revenue per booking 
df_copy["total_revenue"] = df_copy["adr"] * df_copy["guest_total_stay"] 


# <h3 style="color:#edb1f1;">Exporting Dataset</h3>

# In[88]:


df_copy.to_csv(r"c:/users/gautam/downloads/hotel_booking_clean_1.csv",index=False)
df.to_csv(r"c:/users/gautam/downloads/hotel_booking_clean_2.csv",index=False)

