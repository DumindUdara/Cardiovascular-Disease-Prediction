#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[2]:


data = pd.read_csv("/Users/duminduudara/Documents/Study/DS_ML/Machine Learning/data/cardio.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


# Check mising values 
data.isnull().sum()


# In[6]:


data.duplicated().sum()


# In[7]:


data[["age","height","weight","ap_hi","ap_lo"]].boxplot()


# In[8]:


data["age"] = (data["age"]/365).values.astype(int)


# In[9]:


data.head()


# In[10]:


data["bmi"] = data["weight"]/((data["height"]/100)**2)


# In[11]:


data.drop(["height","weight"],axis=1,inplace=True)


# In[12]:


data.head()


# In[17]:


data[["age","ap_hi","ap_lo","bmi"]].dropna().boxplot()
plt.show()


# In[18]:


data_num = data[["age","ap_hi","ap_lo","bmi"]]


# In[19]:


Q1 = data_num.quantile(0.25)
Q3 = data_num.quantile(0.75)

IQR = Q3 - Q1


# In[20]:


IQR


# In[21]:


out_rows = ((data_num<(Q1-1.5*IQR))|(data_num>(Q3+1.5*IQR))).any(axis=1)


# In[23]:


data = data[~out_rows]


# In[24]:


data[["age","ap_hi","ap_lo","bmi"]].boxplot()
plt.show()


# In[25]:


data[["age","ap_hi","ap_lo","bmi"]].describe()


# In[26]:


data["cardio"].value_counts()


# In[31]:


sns.countplot(x="cardio", data=data)
plt.show()


# In[32]:


sns.countplot(data=data,x="age")
plt.show()


# In[33]:


sns.countplot(data=data,x="age",hue="cardio")
plt.show()


# In[36]:


sns.boxplot(x="cardio",y="bmi",data=data)
plt.show()


# In[37]:


sns.countplot(data=data,x="gender",hue="cardio")
plt.show()


# In[40]:


sns.heatmap(data[["age","ap_hi","ap_lo","bmi"]].corr(),annot=True,vmin=-1,vmax=+1)
plt.show()


# In[41]:


sns.pairplot(data[["age","ap_hi","ap_lo","bmi"]])
plt.show()


# In[42]:


df = data[["age","ap_hi","ap_lo","bmi"]]
df["cardio"] = ["Yes" if m==1 else "No" for m in data["cardio"]]
df.head()


# In[45]:


sns.pairplot(df,hue="cardio")
plt.show()


# In[46]:


data.head()


# In[47]:


data.drop("id",axis=1,inplace=True)


# In[48]:


data.head()


# In[49]:


data = data.reindex(columns=["age","ap_hi","ap_lo","bmi","gender","cholesterol","gluc","smoke","alco","active","cardio"])


# In[50]:


data.head()


# In[51]:


# check how many catogories 
data["gender"].value_counts()


# In[52]:


data["cholesterol"].value_counts()


# In[53]:


data["gluc"].value_counts()


# In[54]:


data["smoke"].value_counts()


# In[55]:


data["alco"].value_counts()


# In[57]:


data["active"].value_counts()


# In[58]:


data["cardio"].value_counts()


# In[60]:


le = LabelEncoder()
data["gender"] = le.fit_transform(data["gender"])


# In[61]:


data.head()


# In[63]:


ohe = OneHotEncoder()


# In[73]:


ohot_encode1 = ohe.fit_transform(data["cholesterol"].values.reshape(len(data["cholesterol"].values),1)).toarray()
ohot_encode1 = ohot_encode1[:,1:].astype(int)
df_ohot1 = pd.DataFrame(ohot_encode1,columns=["Above Nor Chol","Well Above Nor Chol"])
df_ohot1


# In[74]:


ohot_encode2 = ohe.fit_transform(data["gluc"].values.reshape(len(data["gluc"].values),1)).toarray()
ohot_encode2 = ohot_encode2[:,1:].astype(int)
df_ohot2 = pd.DataFrame(ohot_encode1,columns=["Above Nor gluc","Well Above Nor gluc"])
df_ohot2


# In[75]:


# Combine these data frame 
data


# In[76]:


data.reset_index(inplace=True)
data


# In[77]:


data.drop("index",axis=1,inplace=True)
data.head()


# In[79]:


x = pd.concat([data.iloc[:,:10],df_ohot1,df_ohot2],axis=1)
x.head()


# In[80]:


x.drop(["cholesterol","gluc"],axis=1,inplace=True)


# In[81]:


x.head()


# In[83]:


# independent dependent 
x = x.values
y = data.iloc[:,10].values


# In[84]:


x


# In[85]:


x[:,:5]


# In[89]:


sc = StandardScaler()


# In[90]:


x[:,:5] = sc.fit_transform(x[:,:5])


# In[91]:


x[:,:5]


# In[86]:


y


# In[93]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[101]:


model1 = KNeighborsClassifier(n_neighbors=50)
model2 = LogisticRegression()
model3 = SVC(kernel="rbf",C=1)
model4 = RandomForestClassifier(n_estimators=500)

T1=('KNNC',model1)
T2=('LR',model2)
T3=('SVM',model3) 
T4=('RFC',model4) 

model = VotingClassifier(estimators=[T1,T2,T3,T4],voting="hard")


# In[102]:


model.fit(x_train,y_train)


# In[103]:


y_pred = model.predict(x_test)


# In[104]:


confusion_matrix(y_test,y_pred)


# In[106]:


accuracy_score(y_test,y_pred)


# In[107]:


print(classification_report(y_test,y_pred))


# In[ ]:




