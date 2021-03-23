#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as ny
import pandas as pd
from matplotlib import pyplot as plt
plt.plot("dataR2.csv")
plt.show("dataR2.csv")
data=pd.read_csv("dataR2.csv")
x=data.drop("Classification",1)
y=data["Classification"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.neural_network import MLPClassifier
model=MLPClassifier(hidden_layer_sizes=(8,8,8),activation="logistic",learning_rate="constant",learning_rate_init=0.1)
model.fit(x_train,y_train)
predict=model.predict(x_test)
from sklearn.metrics import accuracy_score
acc=accuracy_score(predict,y_test)
print(acc)
from sklearn.metrics import confusion_matrix
conf=confusion_matrix(predict,y_test)
print(conf)


# In[ ]:




