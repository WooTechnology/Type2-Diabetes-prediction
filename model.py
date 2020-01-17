#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


pwd


# In[10]:


variable=pd.read_csv(r"C:\Users\aknar\Type2-Diabetes-prediction\Dataset\diabetes.csv")
variable


# In[5]:


#Describe data
variable.describe()


# In[11]:


#information of dataset
variable.info()


# In[7]:


#Check for all null values
variable.isnull().values.any()


# In[8]:


#histogram
variable.hist(bins=10, figsize=(10,10))
plt.show()


# In[9]:


#Correlation
sns.heatmap(variable.corr())
# we see that skin thickness, age, insulin and pregnancies are fully independent on each other
#age and pregnanacies have negative correlation


# In[10]:


#lets count total outcome in each target 0 1
#0 means no diabeted
#1 means patient with diabtes
sns.countplot(y=variable['OUTCOME'],palette='Set1')


# In[11]:


sns.set(style="ticks")
sns.pairplot(variable, hue="OUTCOME")


# In[14]:


#box plot for outlier visualisation
sns.set(style="whitegrid")
variable.boxplot(figsize=(15,6))


# In[12]:


#box plot
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(8,4)})
sns.boxplot(x=variable['INSULIN'])
plt.show()
sns.boxplot(x=variable['BLOOD PRESSURE'])
plt.show()
sns.boxplot(x=variable['DIABETES PEDIGREE FUNCTION'])
plt.show()


# In[14]:


#outlier remove
Q1=variable.quantile(0.25)
Q3=variable.quantile(0.75)
IQR=Q3-Q1
print("---Q1--- \n",Q1)
print("\n---Q3--- \n",Q3)
print("\n---IQR---\n",IQR)

#print((df < (Q1 - 1.5 * IQR))|(df > (Q3 + 1.5 * IQR)))


# In[15]:


#outlier remove
variable_out = variable[~((variable < (Q1 - 1.5 * IQR)) |(variable > (Q3 + 1.5 * IQR))).any(axis=1)]
variable.shape,variable_out.shape
#more than 80 records deleted


# In[16]:


#Scatter matrix after removing outlier
sns.set(style="ticks")
sns.pairplot(variable_out, hue="OUTCOME")
plt.show()


# In[17]:


#lets extract features and targets
X=variable_out.drop(columns=['OUTCOME'])
y=variable_out['OUTCOME']


# In[18]:


#Splitting train test data 80 20 ratio
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)


# In[19]:


train_X.shape,test_X.shape,train_y.shape,test_y.shape


# In[20]:


from sklearn.metrics import confusion_matrix,accuracy_score,make_scorer
from sklearn.model_selection import cross_validate

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

#cross validation purpose
scoring = {'accuracy': make_scorer(accuracy_score),'prec': 'precision'}
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn)}
def display_result(result):
    print("TP: ",result['test_tp'])
    print("TN: ",result['test_tn'])
    print("FN: ",result['test_fn'])
    print("FP: ",result['test_fp'])


# In[21]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

acc=[]
roc=[]

clf=LogisticRegression()
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)

#display predicted values uncomment below line
#pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()


# In[23]:


#Support Vector Machine
from sklearn.svm import SVC

clf=SVC(kernel='linear')
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)

#display predicted values uncomment below line
#pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()


# In[24]:


#KNN

from sklearn.neighbors import KNeighborsClassifier

clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)

#display predicted values uncomment below line
#pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()


# In[ ]:




