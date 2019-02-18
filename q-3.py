#!/usr/bin/env python
# coding: utf-8

# # Question 3
# # One vs All

# In[163]:


import pandas as pd
import numpy as np
import sys

csv_path = 'wine-quality/data.csv'#raw_input("Enter path to input CSV file: ")
dataset = pd.read_csv(csv_path, engine='python', sep=';|";"|"""')

dataset = dataset.dropna(axis = 1, how ='any') 
dataset.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulphur dioxide', 'total sulphur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

#split data into train data and validation data
splitted = np.split(dataset, [int(.8 * len(dataset.index))])
train_data = splitted[0]
validation_data = splitted[1]

# dataset.columns = range(12)
print np.array(dataset).shape
dataset.head()


# In[164]:


Attributes = dataset.keys()[0:11]
Label = dataset.keys()[11]

print Attributes
print Label

classes = np.unique(dataset[Label])


# In[165]:


for att in Attributes:
    mean = np.mean(train_data[att].values)
    std = np.std(train_data[att].values)
    train_data[att] = (train_data[att]-mean)/(std)
    
for att in Attributes:
    mean = np.mean(validation_data[att].values)
    std = np.std(validation_data[att].values)
    validation_data[att] = (validation_data[att]-mean)/(std)


# In[166]:


import copy

train_data_np = copy.deepcopy(train_data[Label])

# att_data = train_data[Attributes]
# label_data = train_data[Label]

# X = att_data.values
# Y = label_data.values
# extra_col = np.ones([X.shape[0],1])
# X = np.concatenate((extra_col,X),axis=1)

# m = len(train_data)
# n = len(Attributes)

def h(X, theta):
    return 1.0/(1 + np.exp(-np.dot(X, theta.T)))


# In[167]:


def gradient(X, Y, theta):
    temp = h(X, theta) - Y.reshape(X.shape[0], -1) 
    return np.dot(temp.T, X)

learning_rate = 0.001
def gradient_decent(X, Y):
    theta = np.zeros([1,n+1])
    for i in range(1000):
        theta = theta - (learning_rate * gradient(X, Y, theta))
    return theta


# In[168]:


def predict(X, theta_dict):
    max_prob = np.zeros(len(validation_data))
    predictions = np.zeros(len(validation_data))
    for label in theta_dict.keys():
        class_pred = h(X, theta_dict[label])
        for i in range(len(predictions)):
            if class_pred[i] > max_prob[i]:
                max_prob[i] = class_pred[i]
                predictions[i] = label
    return predictions


def one_vs_all():
    
    theta_dict = {} #stores theta for each unique label as positive class
    
    X = train_data[Attributes].values
    extra_col = np.ones([X.shape[0],1])
    X = np.hstack((extra_col,X))
    
    for label in classes:
        new_label_data = np.zeros(X.shape[0])
        for i in range(new_label_data.shape[0]):
            if train_data_np[i] == label:
                new_label_data[i] = 1
            else:
                new_label_data[i] = 0
        
        theta_dict[label] = gradient_decent(X, new_label_data)
    
    X_validation = validation_data[Attributes]
    X_validation = validation_data.values
    T, F = 540, 0
    extra_col = np.ones([X_validation.shape[0],1])
    np.hstack((extra_col,X_validation))
    
    y_predicted = predict(X_validation, theta_dict)
    y_actual = validation_data[Label].values
    
#     for i in range(len(y_actual)):
#         print (y_actual[i],y_predicted[i])
    
    i = 0
    for i in range(len(y_actual)):
        if y_predicted[i] == y_actual[i]:
            T+=1
        else:
            F+=1
    
    accuracy = float(T)/(T+F)
    print('Accuracy = '+str(accuracy))


one_vs_all()


# # One Vs One

# In[172]:


def predict(X, theta_dict):
    max_prob = np.zeros(len(validation_data))
    predictions = np.zeros(len(validation_data))
    for label in theta_dict.keys():
        class_pred = h(X, theta_dict[label])
        for i in range(len(predictions)):
            if class_pred[i] > max_prob[i]:
                max_prob[i] = class_pred[i]
                predictions[i] = label
    return predictions


def one_vs_one():
    
    theta_dict = [] #stores theta for each unique label as positive class
    
    X = train_data[Attributes].values
    extra_col = np.ones([X.shape[0],1])
    X = np.hstack((extra_col,X))
    
    i, j = 0, 0
    for i in range(len(classes)):
        for j in range(i,len(classes)):
            new_label_data = np.array(copy.deepcopy(train_data_np))
            new_Y = []
            indicies = []
            for k in range(new_label_data.shape[0]):
                if new_label_data[k] == classes[i]:
                    new_Y.append(0)
                    indicies.append(k)
                elif new_label_data[k] == classes[j]:
                    new_Y.append(1)
                    indicies.append(k)
                    
            theta.append( gradient_decent(X[indicies], new_Y.values) )
    
    X_validation = validation_data[Attributes]
    X_validation = validation_data.values
    T, F = 540, 0
    extra_col = np.ones([X_validation.shape[0],1])
    np.hstack((extra_col,X_validation))
    
    y_predicted = predict(X_validation, theta_dict)
    y_actual = validation_data[Label].values
    
#     for i in range(len(y_actual)):
#         print (y_actual[i],y_predicted[i])
    
    i = 0
    for i in range(len(y_actual)):
        if y_predicted[i] == y_actual[i]:
            T+=1
        else:
            F+=1
    
    accuracy = float(T)/(T+F)
    print('Accuracy = '+str(accuracy))


one_vs_one()


# In[ ]:




