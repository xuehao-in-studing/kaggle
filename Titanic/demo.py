#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2022/12/10 20:14
# @Author : 朱
# @FileName : demo.py

import pandas as pd
from sklearn.naive_bayes import GaussianNB
import sklearn.neighbors
from sklearn.linear_model import LogisticRegression,LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# train_path=r'E:\Program\Python\Kaggle\Titanic\dataset\train.csv'
# test_path=r'E:\Program\Python\Kaggle\Titanic\dataset\test.csv'
#
# train_data=pd.read_csv(train_path)
# test_data=pd.read_csv(test_path)
#
# train_data.head()
# test_data.head()
#
# plt.hist()
#
# print("hello world")

mean=1
std=1
a=std*np.random.randn(1,12)+mean
y=np.random.randn(1,12)
for i in range(6):
    plt.subplot(2,3,i+1)
    sns.scatterplot(a)
plt.show()