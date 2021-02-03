#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 22:55:14 2021

@author: jimlee
"""
import numpy as np
import csv
import sys
from enum import Enum
from numpy.linalg import inv

        
class Method(Enum):
    GENERATIV_MODEL = 'GERNERATIVE_MODEL',
    LOGISTIC_REGRESSION = 'LOGISTIC_REGRESSION'
    
    
class Model():
    def __init__(self):
        self.data = {}
       
    def read(self, name, path): 
        with open(path, newline = '') as csvFile:
            rows = np.array(list(csv.reader(csvFile))[1:], dtype = float)
            
            print(rows.shape[0])
            print(rows.shape[1])
            
            if name == 'X_train':
                self.mean = np.mean(rows, axis = 0)
                self.std = np.std(rows, axis = 0)
                
                for i in range(rows.shape[1]):
                    rows[:,i] = (rows[:,i] - self.mean[i]) / self.std[i]
            
            self.data[name] = rows
            
    def train(self, method):
        if method == Method.GENERATIV_MODEL.value:
            ''' ****** We have to classify the data first ******'''
            class0IndexList = []   
            class1IndexList = []
            
            for i in range(self.data['Y_train'].shape[0]):
                if self.data['Y_train'][i][0] == 0:
                    class0IndexList.append(i)
                else: 
                    class1IndexList.append(i)
                    
            #print(class0IndexList)
            #print(class1IndexList)
            class0 = self.data['X_train'][class0IndexList]
            class1 = self.data['X_train'][class1IndexList]
            
            '''*************************************************'''
            n = class0.shape[1]

            mean0 = np.mean(class0, axis = 0)
            mean1 = np.mean(class1, axis = 0)
            
            cov0 = np.zeros((n,n))
            cov1 = np.zeros((n,n))
            
            for i in range(class0.shape[0]):
                cov0 += np.dot(np.transpose([class0[i] - mean0]), [(class0[i] - mean0)]) / class0.shape[0]

            for i in range(class1.shape[0]):
                cov1 += np.dot(np.transpose([class1[i] - mean1]), [(class1[i] - mean1)]) / class1.shape[0]

            cov = (cov0*class0.shape[0] + cov1*class1.shape[0]) / (class0.shape[0] + class1.shape[0])
            ''' ***** Combine the both covariances *****'''
            print(cov)
            ''' ***** According to the equation, we can calculate our z function as below *****'''
            wT = ((mean0 - mean1).dot(inv(cov))).transpose()
            print(wT)
            b = -0.5 * mean0.transpose().dot(inv(cov)).dot(mean0) + 0.5 * mean1.transpose().dot(inv(cov)).dot(mean1) + class0.shape[0]/class1.shape[0]
            arr = np.empty([self.data['X_train'].shape[0],1], dtype = float)
            for i in range(self.data['X_train'].shape[0]):
                z = wT.dot(self.data['X_train'][i][:]) + b
                z *= -1
                arr[i][0] = 1 / (1 + np.exp(z))
            
            ''' ***** Evaluate the train accuracy ***** '''
            for i in range(arr.shape[0]):
                if arr[i] >= 0.5:
                    arr[i] = 0
                else:
                    arr[i] = 1
            good = 0
            for i in range(arr.shape[0]):
                if arr[i] == self.data['Y_train'][i][0]:
                    good += 1
                else:
                    '''Do Nothing'''
            
            print("Train Accuracy: ", (good / arr.shape[0]) * 100, " %")
        elif method == Method.LOGISTIC_REGRESSION.value:
            ''' wait for edit'''
                
                
                
        
            
            
            
           
            
            
            
  
m = Model()
m.read('X_train', 'X_train')
m.read('Y_train', 'Y_train')
m.train(Method.GENERATIV_MODEL.value)



            
        
    

    