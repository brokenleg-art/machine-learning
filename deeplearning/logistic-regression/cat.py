# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:06:46 2022

@author: xchen
"""

import numpy as np
import matplotlib.pyplot as plt

#import os
#os.chdir(r'D:/onedrive/桌面/deeplearning/deepLearing/cat')
from lr_utils import load_dataset

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

'''
if __name__ == '__main__':
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

#激活函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

#w，b初始化
def init_zeros(dim):
    w = np.zeros(shape = (dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w,b

#一次正向传播，一次反向传播计算
def propagate(w, b, X, Y):
    
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {
            "dw": dw,
            "db": db
        }    
    
    return(grads, cost)

'''
if __name__ == '__main__':
    print("====================测试propagate====================")
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
    grads, cost = propagate(w, b, X, Y)
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    print ("cost = " + str(cost))
'''

#梯度下降迭代算法
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    
    for i in range(num_iterations):
        
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw 
        b = b - learning_rate * db 

        if i % 100 == 0:
            costs.append(cost)
        
        if(print_cost) and (i % 100 == 0):
            print("迭代的次数：{}， 误差值：{}".format(i, cost))
            
    params = {
                "w" : w,
                "b" : b
            }
        
    grads = {
        
            "dw" : dw,
            "db" : db
        
            }
    
    return (params, grads, costs)

'''
if __name__ == '__main__':
    print("====================测试optimize====================")
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
    params , grads , costs = optimize(w , b , X , Y , num_iterations=100 , learning_rate = 0.009 , print_cost = False)
    print ("w = " + str(params["w"]))
    print ("b = " + str(params["b"]))
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
'''

def predict(w, b, X):
    
    #图片的数量
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0,i] > 0.5 else 0
    
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction

'''
if __name__ == '__main__':
#测试predict
    print("====================测试predict====================")
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
    print("predictions = " + str(predict(w, b, X)))
'''

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000 , learning_rate = 0.5 , print_cost = False):
    w, b = init_zeros(X_train.shape[0])
    
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w, b = params["w"], params["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w , b, X_train)
    
    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")
    
    result = {
            "costs" : costs,
             "Y_prediction_test" : Y_prediction_test,
            "Y_prediciton_train" : Y_prediction_train,
            "w" : w,
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations
        
        }
    
    return result

if __name__ == '__main__':
    print("====================测试model====================")     
    #这里加载的是真实的数据，请参见上面的代码部分。
    d = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations = 3000, learning_rate = 0.003, print_cost = True)
    
    costs = np.squeeze(d["costs"])
    plt.plot(costs)
    
    plt.ylabel('costs')
    plt.xlabel('iterations(per hundred)')
    plt.title("learning_rate =" + str(d["learning_rate"]))
    plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    