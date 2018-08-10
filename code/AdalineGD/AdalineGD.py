#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
class AdalineGD(object) :
    '''
    适应性线性神经元分类算法
    eta : float,  学习率, 0~1
    n_iter : int, 迭代次数
    w_ : 权重向量
    cost_ : 神经网络的分类代价
    '''
    
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        pass
    
    def fit(self, X, y) :
        '''神经网络输入训练样本训练模型'''
        # 初始化
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []
        # 训练
        for i in range(self.n_iter) :
            output = self.net_input(X) # 一维向量，长度为2
            errors = y - output  
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()*1
            cost = (errors ** 2).sum() / 2.0 # ** 是python中特有的数学运算符号，表示 x的y次幂运算
            self.cost_.append(cost)
            pass
        return self
        pass
    
    def net_input(self, X) :
        '''神经网路输入函数'''
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass
    
    def activation(self, x):
        '''激活函数'''
        return x
        pass
    
    def predict(self, x):
        '''预测函数，只是对activation函数的输出进行分类判断'''
        return np.where(x>=0.0, 1, -1)
        pass
    pass    
