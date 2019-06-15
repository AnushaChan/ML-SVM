#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 23:15:11 2019

@author: anushachandrasekaran
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss
#from tabulate import tabulate

trn=np.loadtxt("./wdbc_trn.csv", delimiter=',')

#print(len(trn))

y_trn=trn[:, 0]
#print(y_trn)

X_trn=trn[:,1:30]
#print(X_trn)
#print(len(X_trn))

val=np.loadtxt("./wdbc_val.csv", delimiter=',')

y_val=val[:, 0]
X_val=val[:,1:30]

tst=np.loadtxt("./wdbc_tst.csv", delimiter=',')

y_tst=tst[:, 0]
X_tst=tst[:,1:30]

C_range = np.arange(-2.0, 7.0, 1.0)
C_values = np.power(10.0, C_range)
gamma_range = np.arange(-3.0, 6.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnErr = dict()
valErr = dict()
tstErr = dict()

for C_val in C_values:
    for G in gamma_values:
#        print(C_val,G)
        clf = SVC(gamma=G)
        clf.fit(X_trn, y_trn) 
        models[(C_val,G)]=SVC(C=C_val, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=G, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
        models[(C_val,G)].fit(X_trn, y_trn)
        
        #Hinge Loss method
#        pred_decision=models[(C_val,G)].decision_function(X_trn)
#        trnErr[(C_val,G)]=hinge_loss(y_trn, pred_decision)
#    
#        valErr[(C_val,G)]=hinge_loss(y_val, models[(C_val,G)].decision_function(X_val))
#    
#        tstErr[(C_val,G)]=hinge_loss(y_tst, models[(C_val,G)].decision_function(X_tst))
    
#        print((C_val,G),"-",models[(C_val,G)].score(X_tst,y_tst))
        
        trnErr[(C_val,G)]=1-models[(C_val,G)].score(X_trn,y_trn)
        valErr[(C_val,G)]=1-models[(C_val,G)].score(X_val,y_val)
    
    
best_value=min(valErr, key=valErr.get)
print("The best value for C,Gamma from the Training vs Validation error plot is",best_value)
print("The accuracy of the model for C,Gamma =",best_value," is: ", models[best_value].score(X_tst,y_tst))

