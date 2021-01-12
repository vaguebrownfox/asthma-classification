#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 18:22:52 2021

@author: darwin
"""

# libs
#%%
import pandas as pd 
import numpy as np
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

# create folds        
#%%                  
        
def fold(sound_dataframe):
    big = []
    for j in os.listdir(sets_folder):

        file = open(sets_folder + j, 'r') 
        Lines = file.readlines() 

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


# classifiers
#%%

# fitting xgboost
from xgboost import XGBClassifier

def xgb_classifier(train_test_folds) :
    
    xgb_accuracy_test = []
    xgb_accuracy_train = []
    xgb_cm_train = pd.DataFrame();
    xgb_cm_test = pd.DataFrame();
    
    for i in range(0,len(train_test_folds), 2):
        fold_train = train_test_folds[i]
        fold_test = train_test_folds[i + 1]
        
        X = fold_train.iloc[:, :-2]
        y = fold_train.iloc[:, -2]
        
        X_test = fold_test.iloc[:, :-2]
        y_test = fold_test.iloc[:, -2]
        
        classifier_xgb = XGBClassifier(eval_metric="rmsle")
        classifier_xgb.fit(X, y)
        y_pred = classifier_xgb.predict(X_test)

        print("xgb train confusion matrix for fold ", i/2 + 1)
        plot_confusion_matrix(classifier_xgb, X, y)
        cm_train = pd.DataFrame(confusion_matrix(y, classifier_xgb.predict(X), labels=(0, 1)))
        cm_train.append(pd.Series(["0", "0"]), ignore_index=True)
        print(cm_train)
        print("xgb test confusion matrix for fold ", i/2 + 1)
        plot_confusion_matrix(classifier_xgb, X_test, y_test)
        cm_test = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=(0, 1))) 
        print(cm_test)  
        
        xgb_cm_train = pd.DataFrame.append(xgb_cm_train, cm_train)
        
        xgb_cm_test = pd.DataFrame.append(xgb_cm_test, cm_test)
        
        a_test = accuracy_score(y_test, y_pred)
        a_train = accuracy_score(y,classifier_xgb.predict(X))
        
        xgb_accuracy_test.append(a_test)
        xgb_accuracy_train.append(a_train)
        
    result_xgb = {"test_mean_accuracy": np.mean(xgb_accuracy_test),  "test_stdev_accuracy": np.std(xgb_accuracy_test),
                  "train_mean_accuracy": np.mean(xgb_accuracy_train), "train_stdev_accuracy": np.std(xgb_accuracy_train),
                  "cm_test": xgb_cm_test, "cm_train": xgb_cm_train }
    
    return result_xgb

#fitting svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def svm_classifier(train_test_folds):
    
    svm_accuracy_test = []
    svm_accuracy_train = []
    
    svm_cm_train = pd.DataFrame();
    svm_cm_test = pd.DataFrame();
    
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
        
        
        # plot_confusion_matrix(classifier_svm, X_test, y_test)
        print("svm train confusion matrix for fold ", i/2 + 1)
        plot_confusion_matrix(classifier_svm, X, y)
        cm_train = pd.DataFrame(confusion_matrix(y, classifier_svm.predict(X), labels=(0, 1)))
        cm_train.append(pd.Series(["0", "0"]), ignore_index=True)
        print(cm_train)
        print("svm test confusion matrix for fold ", i/2 + 1)
        plot_confusion_matrix(classifier_svm, X_test, y_test)
        cm_test = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=(0, 1))) 
        print(cm_test)  
        
        svm_cm_train = pd.DataFrame.append(svm_cm_train, cm_train)
        
        svm_cm_test = pd.DataFrame.append(svm_cm_test, cm_test)        

        a_test = accuracy_score(y_test, y_pred)
        svm_accuracy_test.append(a_test)
        a_train = accuracy_score(y, classifier_svm.predict(X))
        svm_accuracy_train.append(a_train)

        
    result_svm = {"test_mean_accuracy": np.mean(svm_accuracy_test), "test_stdev_accuracy": np.std(svm_accuracy_test),
                  "train_mean_accuracy": np.mean(svm_accuracy_train),  "train_stdev_accuracy": np.std(svm_accuracy_train),
                  "cm_test": svm_cm_test, "cm_train": svm_cm_train}
    
    return result_svm


# experiments
#%%

# dataset
#%%
sets_folder = "./Sets/"
root_folder = "./data"
data_sets = {}
res_svm = {}
res_xgb = {}
for path, dirs, files in os.walk(root_folder) :
    if len(files) != 0 :
        key = "_".join(list([path.split("/")[i] for i in [2, -1]]))
        dataset = {}
        for file in files :
            dataset[file.split(".")[0]] = fold(pd.read_csv(os.path.join(path, file)))
            res_svm[file.split(".")[0]] = svm_classifier(dataset[file.split(".")[0]])
            res_xgb[file.split(".")[0]] = xgb_classifier(dataset[file.split(".")[0]])
        data_sets[key] = dataset


#%%
result_svm_summary = pd.DataFrame.from_dict(res_svm)

with pd.ExcelWriter('output.xlsx') as writer:  
    result_svm_summary.to_excel(writer, sheet_name='Sheet_name_1')

    # result_svm_summary.to_excel(writer, sheet_name='Sheet_name_2')










