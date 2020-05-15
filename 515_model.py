
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import linear_model
from sklearn import metrics


# In[2]:


bus=pd.read_csv('bus_fe2.csv',header=None)
car=pd.read_csv('car_fe2.csv',header=None)
pickup_truck=pd.read_csv('pickup_truck_fe2.csv',header=None)
articulated_truck=pd.read_csv('articulated_truck_fe2.csv',header=None)
single_unit_truck=pd.read_csv('single_unit_truck_fe2.csv',header=None)
motorcycle=pd.read_csv('motorcycle_fe2.csv',header=None)
van=pd.read_csv('work_van_fe2.csv',header=None)


# In[3]:


bus.info()


# In[4]:


bus.describe()


# In[5]:


bus.head()


# In[6]:


# add labels to each categorial
bus['category']=np.ones([bus.shape[0],1])
car['category']=2*np.ones([car.shape[0],1])
pickup_truck['category']=3*np.ones([pickup_truck.shape[0],1])
articulated_truck['category']=4*np.ones([articulated_truck.shape[0],1])
single_unit_truck['category']=5*np.ones([single_unit_truck.shape[0],1])
motorcycle['category']=6*np.ones([motorcycle.shape[0],1])
van['category']=7*np.ones([van.shape[0],1])


# In[7]:


df=pd.concat([bus,car,pickup_truck,articulated_truck,single_unit_truck,motorcycle,van],ignore_index=True)


# In[8]:


df.info()


# In[9]:


df=df.sample(frac=1).reset_index(drop=True)
df.head()


# In[10]:


x=df.drop(['category'],axis=1)
y=df['category']


# In[11]:


# gaussian process
kernel=1.0*RBF(1.0)
gpc=GaussianProcessClassifier(kernel=kernel,random_state=0).fit(x, y)
gpc.score(x,y)


# In[12]:


x_predict=gpc.predict(x)


# In[13]:


x_predict_prob=gpc.predict_proba(x)


# In[14]:


# multi-class logistic classification
mul_lr=linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg').fit(x,y)
x_predict_log=mul_lr.predict(x)


# In[15]:


x_predict_log


# In[16]:


metrics.accuracy_score(y,x_predict_log)


# In[17]:


lr = linear_model.LogisticRegression().fit(x, y)
metrics.accuracy_score(y,lr.predict(x))


# In[18]:


# cross-validation
train_x, test_x, train_y, test_y = train_test_split(x,y, train_size=0.3)
# gaussian process
kernel=1.0*RBF(1.0)
gpc=GaussianProcessClassifier(kernel=kernel,random_state=0).fit(train_x, train_y)
print(gpc.score(test_x,test_y))
# multi-class logistic classification
mul_lr=linear_model.LogisticRegression(multi_class='multinomial',
                                       solver='newton-cg').fit(train_x,train_y)
print(metrics.accuracy_score(test_y,mul_lr.predict(test_x)))


# In[19]:


lr = linear_model.LogisticRegression().fit(train_x, train_y)
metrics.accuracy_score(test_y,lr.predict(test_x))

