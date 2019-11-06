#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn import datasets
from IPython.display import display, Latex, Markdown
from sklearn.model_selection import train_test_split


# In[6]:


iris = datasets.load_iris()
print(iris)


# In[5]:


iris_frame = pd.DataFrame(iris.data)
# Делаем имена колонок такие же, как имена переменных:
iris_frame.columns = iris.feature_names
# Добавляем столбец с целевой переменной: 
iris_frame['target'] = iris.target
# Для наглядности добавляем столбец с сортами: 
iris_frame['name'] = iris_frame.target.apply(lambda x : iris.target_names[x])
display(iris_frame)


# In[67]:


def dist_sq (a, b) :
    return ((a - b)**2).sum()


def KNN_predict(train_data, test_data, train_labels, n_classes, fitting_iterations):
    gamma = np.zeros(len(train_labels))
    class_indices = [np.argwhere(train_labels == i).ravel() for i in range(n_classes)] #array of arrays of indices of elements of each class
    for i in range(fitting_iterations) :
        for j in range(len(train_data)) :
            classification_values = np.zeros(len(class_indices))
            for cl_ind in range(len(class_indices)):     
                for ind in range(len(class_indices[cl_ind])):
                    classification_values[cl_ind] += np.exp(-0.5 * dist_sq(train_data[j], train_data[class_indices[cl_ind][ind]])) * gamma[class_indices[cl_ind][ind]]

            guess = np.argmax(classification_values)
            if guess != train_labels[j] :
                gamma[j] += 1
    positive_gamma_indices = np.argwhere(gamma).ravel()
    gamma = gamma[positive_gamma_indices]
    train_labels = train_labels[positive_gamma_indices]
    train_data = train_data[positive_gamma_indices]
    class_indices = [np.argwhere(train_labels == i).ravel() for i in range(n_classes)]
    
    predictions = []
    for i in range(len(test_data)):
        cl_values = np.zeros(n_classes)
        for cl_ind in range(len(class_indices)):     
            for ind in range(len(class_indices[cl_ind])):
#                 print(cl_ind, ind, len(class_indices), len(class_indices[cl_ind]), i, len(test_data), len(gamma))
                cl_values[cl_ind] += np.exp(-0.5 * dist_sq(test_data[i], train_data[class_indices[cl_ind][ind]])) * gamma[class_indices[cl_ind][ind]]
            
        predictions.append(np.argmax(cl_values))
            
            
    return predictions


# In[68]:


train_data, test_data, train_labels, test_labels = train_test_split(iris.data, iris.target, test_size=0.33)
# print(train_data, '\n')
# print(test_data, '\n')
# print(train_labels, '\n')
# print(test_labels, '\n')
predictions = KNN_predict(train_data, test_data, train_labels, 3, 20)
print(predictions)
print(test_labels)
print(predictions == test_labels)


# In[25]:


a = [[], [], []]
a[0].append(3)
print(a)


# In[ ]:




