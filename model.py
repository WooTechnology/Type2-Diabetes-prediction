#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pwd


# In[3]:


variable=pd.read_csv(r"C:\Users\aknar\Type2-Diabetes-prediction\Dataset\diabetes.csv")
variable


# In[5]:


#Describe data
variable.describe()


# In[6]:


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


# In[15]:


#box plot
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(8,4)})
sns.boxplot(x=variable['INSULIN'])
plt.show()
sns.boxplot(x=variable['BLOOD PRESSURE'])
plt.show()
sns.boxplot(x=variable['DIABETES PEDIGREE FUNCTION'])
plt.show()


# In[18]:


#outlier remove
Q1=variable.quantile(0.25)
Q3=variable.quantile(0.75)
IQR=Q3-Q1
print("---Q1--- \n",Q1)
print("\n---Q3--- \n",Q3)
print("\n---IQR---\n",IQR)

#print((df < (Q1 - 1.5 * IQR))|(df > (Q3 + 1.5 * IQR)))


# In[19]:


#outlier remove
variable_out = variable[~((variable < (Q1 - 1.5 * IQR)) |(variable > (Q3 + 1.5 * IQR))).any(axis=1)]
variable.shape,variable_out.shape
#more than 80 records deleted


# In[20]:


#Scatter matrix after removing outlier
sns.set(style="ticks")
sns.pairplot(variable_out, hue="OUTCOME")
plt.show()


# In[21]:


#lets extract features and targets
X=variable_out.drop(columns=['OUTCOME'])
y=variable_out['OUTCOME']


# In[22]:


#Splitting train test data 80 20 ratio
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)


# In[23]:


train_X.shape,test_X.shape,train_y.shape,test_y.shape


# In[24]:
# Applying logistic regression model
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
def find_model_perf(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(train_X, train_y)
    y_hat = [x[1] for x in model.predict_proba(test_X)]
    auc = roc_auc_score(y_test, y_hat)
    score = model.score(test_X, test_y)
    
    print("AUC:",auc)
    print("Accuracy Score: ",score)
    
auc_processed = find_model_perf(train_X, train_y, test_X, test_y)
print(auc_processed)

# Result is  as follows:
#AUC: 0.8014950166112957
#Accuracy Score:  0.7421875

# In[25]
# Applying Support Vector Machine
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class SVMModel:
    
    def __init__(self):
        self.classifier = SVC()

    def train(self, train_X, train_y):
        model = self.classifier.fit(train_X, train_y)
        return model
    
    def predict(self, model, test_X):
        return model.predict(test_X)
    
    def evaluate(self, test_y,pred_y, measure):
        if measure=='matrix':
            cm = confusion_matrix(test_y, pred_y , labels=[0, 1])
            return cm
        elif measure=='accuracy':
            return accuracy_score(test_y, pred_y)*100
        else: return None
       
svm = SVMModel()
model = svm.train(train_X, train_y)
predictions = svm.predict(model, test_X)

print (svm.evaluate(test_y, predictions, 'matrix'))
print 
print (svm.evaluate(test_y, predictions, 'accuracy'))

#Result:
#[[87  0]
#[41  0]]
#67.96875

#KNN Model:
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import neighbors, preprocessing

from sklearn import metrics

knn = neighbors.KNeighborsClassifier()
knn.fit(train_X, train_y)

accuracy = knn.score(test_X, test_y)
prediction = knn.predict(test_X)
accuracy

#Result:
#0.734375
