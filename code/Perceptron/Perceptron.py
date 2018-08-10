#! /usr/bin/python3
# -*- code: utf-8 -*-

import numpy as np
class Perceptron(object):
    """
    感知器类，表示单个神经元
    eta : 感知器学习率
    n_iter : 迭代次数
    w_ : 权重向量数组
    errors_ : 错误次数数组, 用于记录神经元判断错误次数
    """
    
    def __init__(self, eta = 0.01, n_iter = 10) :
        '''构造函数'''
        self.eta = eta
        self.n_iter = n_iter
        pass
    
    def fit(self, X, y):
        """
        输入训练数据，培训神经元
        X： 输入数据样本
        y： 输入数据样本的对应分类
        X: shape[n.samples, n_features] numpy中的函数，表示多维数组的属性
        eg :
        X: [[1,2,3], [4,5,6]]
        n_samples : 2, n_feature : 3
        y : [1, -1]
        """
        
        '''初始化基本数据'''
        self.w_  = np.zeros(1+X.shape[1]) # 加1是为w0, 表示阈值
        self.errors_ = [] # 初始化为数组
        
        '''开始训练'''
        for i in range(self.n_iter) :
            errors = 0
            for xi, target in zip(X,y) : # 每一行样本数据以及它的分类
                '''
                输入样本数据
                xi : 样本数据行
                y : 样本数据对应分类
                '''
                z = self.net_input(xi)
                '''激活函数'''
                _y = self.predict(z)
                '''调整权重向量'''
                update = self.eta * (target - _y)
                self.w_[1:] += update * xi # w[i] = eta * (y - _y) * xi
                self.w_[0] = update
                errors += int(update != 0.0)
                pass
            self.errors_.append(errors)
            pass
        pass

    def net_input(self, X) :
        '''每一行样本数据输入得到输出结果'''
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass
    
    def predict(self, z):
        '''激活函数，只是根据net_input函数的输出结果判断分类'''
        return np.where(z >= 0.0, 1, -1)
        pass
    
    pass

