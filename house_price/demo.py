#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2022/12/13 9:33
# @Author : 朱
# @FileName : demo.py

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from lazypredict.Supervised import LazyRegressor
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split
# 显示中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

train_path = r'E:\Program\Python\Kaggle\house_price\dataset\train.csv'
test_path = r'E:\Program\Python\Kaggle\house_price\dataset\test.csv'

# 训练数据
data = pd.read_csv(train_path)
# 提交结果的数据
test_data = pd.read_csv(test_path)

data.head()
data.info()

# 数值化的列
# columns_numerical = {i: data[i].dtype for i in data.columns if data[i].dtype != object}
# 更简便的写法
columns_numerical = (data.dtypes[data.dtypes!=object]).index
columns_object = {i: data[i].dtype for i in data.columns if data[i].dtype == object}

data['SalePrice'].describe()


for j in columns_numerical:
    plt.scatter(data[j], data['SalePrice'], label=j)
    plt.title(f'SalePrice与{j}的散点图')
    plt.xlabel(j)
    plt.ylabel('SalePrice')
    plt.legend()
    plt.show()

var = 'OverallQual'
data_box = pd.concat([data[var], data['SalePrice']], axis=1)
plt.figure(figsize=(16, 8))
sns.boxplot(x=var, y='SalePrice', data=data_box)
plt.xlabel(var)
plt.ylabel('SalePrice')
plt.xticks(rotation=90)
plt.show()

var = 'YearBuilt'
data_box = pd.concat([data[var], data['SalePrice']], axis=1)
plt.figure(figsize=(16, 8))
sns.boxplot(x=var, y='SalePrice', data=data_box)
plt.xlabel(var)
plt.ylabel('SalePrice')
plt.xticks(rotation=90)
plt.show()

corr = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, square=True, annot=False)
plt.show()

# saleprice correlation matrix
k = 10  # number of variables for heatmap
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
corr_10 = corr.loc[cols, cols]

plt.figure(figsize=(12, 8))
sns.heatmap(corr_10, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
            xticklabels=cols.values)
plt.show()

data_copy = data[set(cols) - {'1stFlrSF', 'GarageCars', 'TotRmsAbvGrd'}]
plt.figure(figsize=(12, 8))
sns.pairplot(data_copy, size=2.5)
plt.show()

total = data.isnull().sum().sort_values(ascending=False)
percentage = data.isnull().sum().sort_values(ascending=False) / data.isnull().count()
percentage.sort_values(ascending=False, inplace=True)
null = pd.concat([total, percentage], axis=1, keys=['total', 'percentage'])

# 剔除缺失值
data.drop(null[null['total'] > 1].index, axis=1, inplace=True)
data.drop(data[data['Electrical'].isnull()].index, inplace=True)
data.isnull().sum().sum()

sns.distplot(data['TotalBsmtSF'], fit=norm)
plt.figure()
res = stats.probplot(data_copy['TotalBsmtSF'], plot=plt)
plt.show()

standard_toll = StandardScaler()
price_scaled = standard_toll.fit_transform(data['SalePrice'][:, np.newaxis])
# 较小的值和较大的值
print(price_scaled[np.argsort(data['SalePrice'])][:10, ])  # 升序排列
print(price_scaled[np.argsort(data['SalePrice'])][-10:, ])  # 升序排列

data[['SalePrice', 'GrLivArea']].plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
plt.show()

# 去除掉不合理的'GrLivArea'
data.drop(data['GrLivArea'].sort_values(ascending=False)[:2].index, inplace=True)
data[['SalePrice', 'GrLivArea']].plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
plt.show()

data_copy = data.copy(deep=True)

plot_columns = set(cols) - {'1stFlrSF', 'GarageCars', 'TotRmsAbvGrd', 'SalePrice'}
for index, column in enumerate(plot_columns):
    ax = plt.subplot(2, 3, index + 1)
    # 以正态分布作基准
    sns.distplot(data[column], fit=norm)
plt.figure(figsize=(12, 4))
plt.show()
# qq图

plt.subplot(121)
sns.distplot(data['SalePrice'], fit=norm)
# qq图
plt.subplot(122)
res = stats.probplot(data['SalePrice'], plot=plt)
plt.show()

# 对数转换对于正偏态很有作用
data_copy['SalePrice'] = np.log(data['SalePrice'])
# 再次作图
plt.subplot(121)
sns.distplot(data_copy['SalePrice'], fit=norm)
# qq图
plt.subplot(122)
res = stats.probplot(data_copy['SalePrice'], plot=plt)
plt.show()

# 划分数据集
train_data = data_copy[set(corr_10) - {'1stFlrSF', 'GarageCars'}]
sc_x = StandardScaler()
sc_y = StandardScaler()
# sc.fit(train_data)
# train_data=sc.fit_transform(train_data)

X = train_data[set(train_data.columns) - set('SalePrice')]
y = train_data['SalePrice']

X, y = sc_x.fit_transform(X), sc_y.fit_transform(y.values.reshape((-1, 1)))

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
reg = LazyRegressor()
model, predictions = reg.fit(train_x, test_x, train_y, test_y)

temp_df = model.sort_values('RMSE', ascending=True)[:-2]
sns.barplot(x=temp_df['RMSE'], y=temp_df.index).set_title('models based on RMSE')

from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor()
mlp.fit(X, y)

test = test_data[set(corr_10) - {'1stFlrSF', 'GarageCars', 'SalePrice'}]
print(f"缺失值的数量{test.isnull().sum().sum()}")

# 去除缺失值
test.fillna(0, inplace=True)
print(f"缺失值的数量{test.isnull().sum().sum()}")
sc_x.fit_transform(test)
# 预测
test = sc_x.fit_transform(test)
outcome = mlp.predict(test)

std = np.sqrt(sc_y.var_)
mean = sc_y.mean_
# 原始形式
outcome = outcome * std + mean
outcome = np.exp(outcome)
outcome = pd.DataFrame({'Id': test_data.Id, 'SalePrice': outcome})
outcome.to_csv('submission.csv', index=False)
