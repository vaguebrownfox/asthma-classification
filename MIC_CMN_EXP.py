#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:25:33 2021

@author: darwin
"""

# libs
import pandas as pd 
import numpy as np
import os

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')


# dataset
mic_data = pd.read_csv("./withoutCMN-12X53mic.csv")
mic_data_cmn = pd.read_csv("./withCMN-12X53mic.csv")

# generate folds - #7
def fold(sound_dataframe, sets_loc):
    big = []
    for j in os.listdir(sets_loc):

        file = open(sets_loc+j, 'r') 
        Lines = file.readlines() 

        count = 0
        try:
            set_dataframe = pd.DataFrame()
            for i, line in enumerate(Lines):
                A = sound_dataframe[(sound_dataframe['73'] == (Lines[i].strip()+".json"))]
                set_dataframe = pd.DataFrame.append(set_dataframe,A)
        except:
            set_dataframe = pd.DataFrame()
            for i, line in enumerate(Lines):
                A = sound_dataframe[(sound_dataframe[73] == (Lines[i].strip()+".json"))]
                set_dataframe = pd.DataFrame.append(set_dataframe,A)
            
        
        big.append(set_dataframe)
    fold_1 = shuffle(pd.concat(big[:6]))
    test_1 = shuffle(big[6])
    fold_2 = shuffle(pd.concat(big[1:7]))
    test_2 = shuffle(big[0])
    fold_3 = shuffle(pd.concat([big[0],big[2],big[3],big[4], big[5], big[6]]))
    test_3 = shuffle(big[1])
    fold_4 = shuffle(pd.concat([big[0],big[1],big[3],big[4], big[5], big[6]]))
    test_4 = shuffle(big[2])
    fold_5 = shuffle(pd.concat([big[0],big[2],big[1],big[4], big[5], big[6]]))
    test_5 = shuffle(big[3])
    fold_6 = shuffle(pd.concat([big[0],big[2],big[1],big[3], big[5], big[6]]))
    test_6 = shuffle(big[4])
    fold_7 = shuffle(pd.concat([big[0],big[2],big[1],big[4], big[3], big[6]]))
    test_7 = shuffle(big[5])
    
    return [fold_1,test_1,fold_2,test_2,fold_3,test_3,fold_4,test_4,fold_5,test_5, fold_6, test_6, fold_7,test_7]


sets_path = './Sets/'
mic_fold_list = fold(mic_data, sets_path)
mic_cmn_fold_list = fold(mic_data_cmn, sets_path)


# fitting xgboost
from xgboost import XGBClassifier

def xgb_classifier(train_test_folds) :
    
    xgb_accuracy_test = []
    xgb_accuracy_train = []
    
    for i in range(0,len(train_test_folds), 2):
        fold_train = train_test_folds[i]
        fold_test = train_test_folds[i + 1]
        
        X = fold_train.iloc[:, :-2]
        y = fold_train.iloc[:, -2]
        
        X_test = fold_test.iloc[:, :-2]
        y_test = fold_test.iloc[:, -2]
        
        classifier_xgb = XGBClassifier()
        classifier_xgb.fit(X, y)
        y_pred = classifier_xgb.predict(X_test)
        
        a_test = accuracy_score(y_test, y_pred)
        xgb_accuracy_test.append(a_test)
        a_train = accuracy_score(y,classifier_xgb.predict(X))
        xgb_accuracy_train.append(a_train)
        
    mean_accuracy_xgb = {"test_mean_accuracy": np.mean(xgb_accuracy_test), 
                         "train_mean_accuracy": np.mean(xgb_accuracy_train)}
    
    return mean_accuracy_xgb

#fitting svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def svm_classifier(train_test_folds):
    
    svm_accuracy_test = []
    svm_accuracy_train = []
    
    for i in range(0,len(train_test_folds),2):
        fold_train = train_test_folds[i]
        fold_test = train_test_folds[i + 1]
        
        X = fold_train.iloc[:, :-2]
        y = fold_train.iloc[:, -2]
        
        X_test = fold_test.iloc[:, :-2]
        y_test = fold_test.iloc[:, -2]


        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)

        X = pd.DataFrame(X)
        X_test = pd.DataFrame(X_test)

        classifier_svm = SVC()
        classifier_svm.fit(X, y)
        y_pred = classifier_svm.predict(X_test)


        a_test = accuracy_score(y_test, y_pred)
        svm_accuracy_test.append(a_test)
        a_train = accuracy_score(y, classifier_svm.predict(X))
        svm_accuracy_train.append(a_train)
        
    mean_accuracy_svm = {"test_mean_accuracy": np.mean(svm_accuracy_test), 
                         "train_mean_accuracy": np.mean(svm_accuracy_train)}
    
    return mean_accuracy_svm


def evaluate():
    res = {"mic": {"xgb": xgb_classifier(mic_fold_list), "svm": svm_classifier(mic_fold_list)},
         "mic_cmn": {"xgb": xgb_classifier(mic_cmn_fold_list), "svm": svm_classifier(mic_cmn_fold_list)}}

    return res
















