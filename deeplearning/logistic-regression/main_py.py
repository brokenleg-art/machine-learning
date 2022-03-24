# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:50:32 2022

@author: xchen
"""

import cat as c
import numpy as np
import matplotlib.pyplot as plt

from lr_utils import load_dataset
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

'''
测试代码
index = 60
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y_orig[:, index]) + ", it's a " + classes[np.squeeze(train_set_y_orig[:,index])].decode("utf-8") + " picture")
'''

#把图片数据转化为一维数据：（64*64*3，1）
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

#print(str(train_x_flatten.shape))
#数据归一化
train_set_x = train_x_flatten / 255
test_set_x = test_x_flatten / 255

#print(str(train_set_y_orig.shape))

d = c.model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations = 3000, learning_rate = 0.003, print_cost = True)

costs = np.squeeze(d["costs"])
plt.plot(costs)

plt.ylabel('costs')
plt.xlabel('iterations(per hundred)')
plt.title("learning_rate =" + str(d["learning_rate"]))
plt.show()
