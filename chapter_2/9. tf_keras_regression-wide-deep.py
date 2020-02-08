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


# In[14]:


# 函数式API 功能API
input = keras.layers.Input(shape=x_train.shape[1:])
# 实现deep
hidden1 = keras.layers.Dense(30,activation="sigmoid")(input)
hidden2 = keras.layers.Dense(30,activation="sigmoid")(hidden1)
# 复合函数：f(x) = g(g(x))

# wide 和 deep 合并
concat = keras.layers.concatenate([input,hidden2])

# 输出
output = keras.layers.Dense(1)(concat)

# 固化模型
model = keras.models.Model(inputs=[input],outputs = [output])


# 打印model信息
model.summary()
# 编译
model.compile(loss='mean_squared_error',optimizer="adam")
# 回调函数
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]


# In[15]:


#训练
history = model.fit(x_train_scaled,y_train,validation_data=(x_valid_scaled,y_valid),epochs=100,callbacks=callbacks)


# In[16]:


# 学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)


# In[17]:


model.evaluate(x_test_scaled,y_test)


# In[ ]:




