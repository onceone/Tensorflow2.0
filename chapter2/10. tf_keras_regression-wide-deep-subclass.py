#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl,np,pd,sklearn,tf,keras:
    print(module.__name__,module.__version__)


# In[2]:


from sklearn.datasets import fetch_california_housing

# 房价预测
housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)


# In[3]:


import pprint

pprint.pprint(housing.data[0:5])
pprint.pprint(housing.target[0:5])


# In[4]:


# 划分样本
from sklearn.model_selection import train_test_split

x_train_all,x_test,y_train_all,y_test = train_test_split(housing.data,housing.target,random_state=7)
x_train,x_valid,y_train,y_valid = train_test_split(x_train_all,y_train_all,random_state=11)

print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)
print(x_test.shape,y_test.shape)


# In[5]:


# 归一化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)


# In[11]:


# 子类API
class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel,self).__init__()
        """定义模型层次"""
        self.hidden1_layer = keras.layers.Dense(30,activation='relu')
        self.hidden2_layer = keras.layers.Dense(30,activation='relu')
        self.output_layer = keras.layers.Dense(1)
    
    def call(self,input):
        """完成模型的正向计算"""
        hidden1 = self.hidden1_layer(input)
        hidden2 = self.hidden2_layer(hidden1)
        concat = keras.layers.concatenate([input,hidden2])
        output = self.output_layer(concat)
        
        return output
    
# model = WideDeepModel()
# 另一种方法
model = keras.models.Sequential([
    WideDeepModel(),
])


model.build(input_shape=(None,8))

# 打印model信息
model.summary()
# 编译
model.compile(loss='mean_squared_error',optimizer="adam")
# 回调函数
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]


# In[12]:


#训练
history = model.fit(x_train_scaled,y_train,validation_data=(x_valid_scaled,y_valid),epochs=100,callbacks=callbacks)


# In[8]:


# 学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)


# In[14]:


plot_learning_curves(history)


# In[10]:


model.evaluate(x_test_scaled,y_test)


# In[ ]:




