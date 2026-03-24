#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-family:fantasy; font-weight:500; font-size:4em; color:blue">Model Building</h1>

# In[55]:


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score , accuracy_score ,f1_score, precision_score, recall_score , roc_curve
import joblib 


# In[2]:


# import dataset using encoded dataset
df = pd.read_csv("../Datasets/hotel_df_with_encoded.csv")


# In[3]:


# data head
df.head()


# In[4]:


# Feature that will be used to avoid any conflict with target leakage
col = ["hotel", 'lead_time' , "arrival_date_year" , "arrival_date_month" , 
       "arrival_date_week_number","arrival_date_day_of_month" , "stays_in_weekend_nights",
       "stays_in_week_nights", "adults", "children", "babies", 'meal', "country",
       'market_segment', 'distribution_channel', 'is_repeated_guest','previous_cancellations',
       'previous_bookings_not_canceled','reserved_room_type' 
       , 'booking_changes', 'deposit_type', 'agent', 'company', 'days_in_waiting_list', 
       'customer_type', 'adr', 'required_car_parking_spaces', 'total_of_special_requests', 
       'guest_total_stay', "is_stay_long"]


# In[5]:


# Feature Normalization
x = df[col] # independent variable
y = df["is_canceled"] # dependent variable
std = StandardScaler()
std_x = std.fit_transform(x) # normalizing dataset 


# In[6]:


# reviewing normalized data
std_x


# In[7]:


# splitting data into train and test set 
x_train , x_test , y_train, y_test = train_test_split(std_x, y, random_state=21, test_size=0.7)
# review shape
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# dataset is split in ratio of 70% data goes for train purpose and 30% is for testing purpose 


# In[33]:


# model building 
Model = RandomForestClassifier(n_estimators=250,class_weight="balanced", min_samples_leaf=10, max_depth=30) 
# random forest classifier model with n_estimators 250 and class will be set to balanced , min_samples_leaf =10 , max depth 30
# fitting data into model or train model
Model.fit(x_train, y_train)
# predicting x test 
pred = Model.predict(x_test)


# In[34]:


# score's and accuracy 
print(f"Roc_auc : {roc_auc_score(pred , y_test)}")
print(f"Accuracy Score : {accuracy_score(pred,y_test)}")
print(f"F1_Score : {f1_score(pred, y_test)}")
print(f"Precision Score: {precision_score(pred, y_test)}")
print(f"Recall Score: {recall_score(pred,y_test)}")


# In[54]:


# plotting roc curve
y_scores= Model.predict_proba(x_test)[:,1] # getting y scores 
fpr , tpr , threshold = roc_curve(y_test, y_scores) # getting false positive rate , true positive rate
roc_auc = roc_auc_score(y_test, y_scores) # getting roc auc score which is turned out be approx 90% of the well sepration
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='red', linestyle='--', lw=2, label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[57]:


# Saving model
joblib.dump(Model, "Hotel_Booking_Model.pkl")


# In[ ]:




