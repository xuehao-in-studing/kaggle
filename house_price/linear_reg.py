#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2023/1/3 16:32
# @Author : 朱
# @FileName : linear_reg.py

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l
from sklearn.model_selection import cross_validate
import numpy as np
from torch.utils.data import DataLoader


train_path = r'E:\Program\Python\Kaggle\house_price\dataset\train.csv'
test_path = r'E:\Program\Python\Kaggle\house_price\dataset\test.csv'

# 读入数据
train_data=pd.read_csv(train_path)
test_data=pd.read_csv(test_path)

# 查看数据
train_data.head()

# 数据集制作
train_features=train_data.iloc[:,1:-1]
train_labels=train_data.iloc[:,-1]
test_features=test_data.iloc[:,1:]

# 数值化的列
numerical=(train_features.dtypes[train_features.dtypes!=object]).index
train_features[numerical]=train_features[numerical].apply(lambda x:(x-x.mean())/x.std())
test_features[numerical]=test_features[numerical].apply(lambda x:(x-x.mean())/x.std())
print(f'缺失值处理前，缺失值数量{train_features.isnull().sum().sum()}')

# 简单用均值替代所有缺失值
train_features.fillna(0,inplace=True)
test_features.fillna(0,inplace=True)
print(f'缺失值处理后，缺失值数量{train_features.isnull().sum().sum()}')


train_features=pd.get_dummies(train_features)
test_features=pd.get_dummies(test_features)
print(train_features.head())

# 转换为张量
train_features=torch.tensor(train_features.values,dtype=torch.float32)
test_features=torch.tensor(test_features.values,dtype=torch.float32)
train_labels = torch.tensor(
    train_labels.values.reshape(-1, 1), dtype=torch.float32)

input_feature=train_features.shape[-1]
# 定义模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=nn.Sequential(nn.Linear(input_feature,1))
net.to(device)

loss=nn.MSELoss()
loss.to(device)
train_ls, test_ls = [], []

# def log_mse(features,labels):
#     rmse = torch.sqrt(loss(torch.log(features),
#                            torch.log(labels)))
#     return rmse.item()


def train(net,train_features,train_labels,loss,
          num_epochs,lr,weight_decay,batch_size):
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            X,y=X.to(device),y.to(device)
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(loss(train_features, train_labels))
        print(f'info:epoch is {epoch},train_loss is {train_ls[-1]}')

    plt.plot(range(num_epochs),train_ls,label='train_ls')
    plt.legend()
    plt.grid()
    plt.show()

def pred(net,test_features):
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    num_epochs,lr,weight_decay,batch_size=20,1e-3,5,128
    train(net,train_features,train_labels,loss,num_epochs,lr,weight_decay,batch_size)
    pred(net,test_features)
