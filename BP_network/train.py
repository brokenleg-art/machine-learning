# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:52:47 2022

@author: xchen
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
from neuNet import NeuralNetwork
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

#读取数据，拆分训练集，测试集
datafile = 'D:/onedrive/桌面/bp_net/bp_network/dataset_2.xlsx'
data = pd.read_excel(datafile)
exam = DataFrame(data)

new_exam=exam.loc[:,['width','depth','ved','awz']]
#new_exam.head

nn = NeuralNetwork([3, 100, 6], 'logistic')
x_train,x_test,y_train,y_test = train_test_split(new_exam.loc[:,['width','depth','ved']],new_exam.loc[:,['awz']],train_size=0.8)
#正交化1表达
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)

print("start fitting...")
nn.fit(x_train, labels_train, 3000)
predictions = nn.predict(x_test)
print(classification_report(y_test, predictions))

 
