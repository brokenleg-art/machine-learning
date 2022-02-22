# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
class NeuralNetwork:
    #生长函数及其导数
    def logistic(self,x):
        return 1/(1 + np.exp(-x))
    def logistic_derivative(self,x):
        return self.logistic(x)*(1 - self.logistic(x))
    
    def tanh(self,x):
        return np.tanh(x)
    def tanh_derivative(self,x):
        return 1.0 - np.tanh(x)**2
    
    #初始化，【10，5，2】表示三层神经元及每层的神经元个数
    def __init__(self,layers,activation='tanh'):
        
        if activation == 'logistic':
            self.activation = self.logistic
            self.activation_deriv = self.logistic_derivative
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_deriv = self.tanh_derivative
        
        self.weight = []
    
    #确定初始权重
        for i in range(len(layers) - 1):
            if i == 0:
                w = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
                self.weight.append(w * 0.5)        
            else:
                w = 2 * np.random.random((layers[i], layers[i + 1])) - 1
                self.weight.append(w * 0.5)
    
    #训练函数            
    def fit(self, X, y, learning_rate = 0.2, epochs = 10000):
        X = np.atleast_2d(X)
        temp = np.ones(X.shape[0])
        
        X=np.c_[X,temp]
        for k in range(epochs):
            
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            
            #完成所有正向的更新
            for j in range(len(self.weight)):
                a.append(self.activation(np.dot(a[j], self.weight[j])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]
                
            #反向误差传播
            for j in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weight[j].T)*self.activation_deriv(a[j]))
            deltas.reverse()
            
            for i in range(len(self.weight)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weight[i] += learning_rate * layer.T.dot(delta)
    
    #预测函数
    def predict(self,x):
        temp = np.ones(x.shape[0])
        x = np.c_[x,temp]
        ans = []
        for a in x:
            for w in self.weight:
                a = self.activation(np.dot(a,w))
            ans.append(np.argmax(a))
            #ans.append(a[0])
        return ans
    