# -*- coding: utf-8 -*-
import matplotlib as mpl
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
###################
import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


sample_len=693
count_tot = [0]*2 #Number of samples per category
count_test = [0]*2#Number of test samples per category
count_train = [0]*2  #Number of training samples per category
a = np.array([]) #data
b = np.array([]) #label
data_count = 0 #numbers of all data
data_filepath = [] #used to output the path of misclassification samples
test_filepath = []
feature_name = np.array([])

class_label = [0,1]
cm=np.zeros((len(class_label),len(class_label)))
class_name = ['nap','non']
train_name = ['txt0']
groups = []    #training set: 1,  test set: 0
iamtrain = 0 

####-------------------------------------------------------------------------------------------------------------------
def gci(filepath,i):
    global class_label
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        if os.path.isdir(fi_d):
            for k in range(len(class_name)):
                 if(fi_d.endswith(class_name[k])):
                    i = class_label[k]
            gci(fi_d,i)
        else:
            if fi_d.endswith(".txt") == 1:
                readFile(os.path.join(filepath,fi_d),i)


####-------------------------------------------------------------------------------------------------------------------
#spectra processing
def derivative(data):
    result = [data[i] - data[i - 1] for i in range(1, len(data))]
    result.insert(0, result[0])
    return result

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
####-------------------------------------------------------------------------------------------------------------------
# load
def readFile(filename,i):
    if (i == -1): 
        return
    global a
    global b
    global data_count
    global groups
    global iamtrain
    global data_filepath
    global feature_name
    global test_name_all
    count_tot[i] = count_tot[i] + 1
    c = np.loadtxt(filename, dtype=float, usecols=1, unpack=True)
    feature_name = np.loadtxt(filename, dtype=float, usecols=0, unpack=True)
    c = np.array(c)
    xx = max(c)
    c = [x/xx for x in c]
    c = derivative(c) 
    b = np.concatenate((b, np.array([i])), axis=0)
    is_train = 0
    for trainname in train_name:
        if filename.find(trainname) != -1:
            is_train = 1
            iamtrain = data_count
            break
    groups = groups + [is_train]
    if is_train == 1:
        count_train[i] = count_train[i] + 1 
    else:
        count_test[i] = count_test[i] + 1
    a = np.concatenate((a,c),axis=0)
    data_count = data_count + 1
    data_filepath = data_filepath + [filename]
####-------------------------------------------------------------------------------------------------------------------

####-------------------------------------------------------------------------------------------------------------------
#confusion matrix
def plot_confusion_matrix(cm, classes, figpath, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)
        #cm = 'cm.astype / cm.sum'[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(figpath, dpi=500)
    plt.show()
####-------------------------------------------------------------------------------------------------------------------
def num2color(values, cmap):
    """Map values to colors"""
    norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
    cmap = mpl.cm.get_cmap(cmap)
    return [cmap(norm(val)) for val in values]

####-------------------------------------------------------------------------------------------------------------------
def gcf(X_train, X_test, y_train, y_test):

    global test_filepath
    train_num = len(y_train)
    test_num = len(y_test)
    global class_name
    global count_tot
    global count_train
    global count_test
    global index_See
    global cm
    index_See=[]
    class_name_s = 'class_name: '
    for k in range(len(class_label)):
        class_name_s = class_name_s + str(class_name[k]) + ' '
    class_name_s = class_name_s + '\n'
    count_tot_s = 'count_tot: '
    for k in range(len(class_label)):
        count_tot_s = count_tot_s + str(count_tot[k]) + ' '
    count_tot_s = count_tot_s + '\n'
    count_train_s = 'count_train: '
    for k in range(len(class_label)):
        count_train_s = count_train_s + str(count_train[k]) + ' '
    count_train_s = count_train_s + '\n'
    count_test_s = 'count_test: '
    for k in range(len(class_label)):
        count_test_s = count_test_s + str(count_test[k]) + ' '
    count_test_s = count_test_s + '\n'
    train_system_s = 'train_system: '
    for k in range(len(train_name)):
        train_system_s = train_system_s + train_name[k] + ' '
    train_system_s = train_system_s + '\n'
    repeat_times = 10 
    param_num = 1
    param_trees = [1000]
    i = 0
    while i < param_num:  
        result_f_name = 'C:/Users/lgkgroup/Desktop/test2/param_trees' +str(param_trees[i]) + '.txt'##################################
        result_f = open( result_f_name, 'w')
        result_f.write('train_num: ' + str(train_num) + '    test_num: ' + str(test_num) +'\n\n')
        result_f.write(class_name_s)
        result_f.write(count_tot_s)
        result_f.write(count_train_s)
        result_f.write(count_test_s)
        result_f.write(train_system_s)
        result_f.write('params: trees = ' + str(param_trees[i]) + '\n')
        accuracy_tot = 0
        accuracy_best = 0
        accuracy_worst = 1.0
        oob_score_tot=0
        oob_score_best=0
        oob_score_worst=1
        time_tot = 0
        time_best = 10000
        time_worst = 0
        counter = 1
        precision_all=[]
        oob_precision_all=[]
        total=np.zeros((len(y_test),2))
        while counter <= repeat_times:
            print('\n----- count:', counter, ' -----')      
            start = time.clock()
            rf_clf = RandomForestClassifier(n_estimators=param_trees[i],max_features='auto',n_jobs=-1,oob_score=True)
            rf_clf.fit(X_train, y_train)
            y_pred = rf_clf.predict(X_test)
            total = total+rf_clf.predict_proba(X_test)
            print(rf_clf.predict_proba(X_test))
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            figpath='C:/Users/lgkgroup/Desktop/test2/'+str(counter) +'Confusion matrix.png'
            plot_confusion_matrix(cnf_matrix, classes=class_name,figpath=figpath,title='Confusion matrix')
            end = time.clock()
            time_temp = end - start
            print('time: ', time_temp)
            time_tot += time_temp
            if time_temp > time_worst:
                time_worst = time_temp
            if time_temp < time_best:
                time_best = time_temp
            importance_all=np.zeros(sample_len,)
            importance_all=importance_all+rf_clf.feature_importances_#################################################
            accuracy_temp = metrics.accuracy_score(y_test, y_pred)
            print()
            precision_all.append(accuracy_temp)
            print('accuracy:', accuracy_temp)
            print(classification_report(y_test, y_pred, target_names=class_name))
            cm = cm+cnf_matrix
            accuracy_tot += accuracy_temp
            if accuracy_temp > accuracy_best:
                accuracy_best = accuracy_temp
            if accuracy_temp < accuracy_worst:
                accuracy_worst = accuracy_temp
            oob_score_temp=rf_clf.oob_score_
            oob_precision_all.append(oob_score_temp)
            oob_score_tot+=oob_score_temp
            print('OBB:',oob_score_temp)
            if oob_score_temp > oob_score_best:
                oob_score_best = oob_score_temp
            if oob_score_temp < oob_score_worst:
                oob_score_worst = oob_score_temp
            importance=map(lambda x: round(x, 7), rf_clf.feature_importances_)
            print(list(zip(feature_name,importance)))
            result_f.write('\n----- count:' + str(counter) + ' -----\n')
            result_f.write("time: " + str(time_temp)+ '\n')
            result_f.write('accuracy: ' + str(accuracy_temp) + '\n')
            for k in range(test_num):
                if y_test[k] != y_pred[k]:
                    result_f.write(str(class_name[int(y_test[k])]) + ' -> ' + str(class_name[int(y_pred[k])]) + ' ' + test_filepath[k] + '\n\n')
            counter = counter + 1
        xx = max(importance_all)
        importance_ = [x/xx for x in importance_all]
        np.savetxt('C:/Users/lgkgroup/Desktop/test2/total' +str(param_trees[i]) + '.txt', total)
        precision= open('C:/Users/lgkgroup/Desktop/test2/precision' +str(param_trees[i]) + '.txt', 'w')
        for k in precision_all:
            precision.write(str(k) + "\n")
        precision.close()
        oob_precision= open('C:/Users/lgkgroup/Desktop/test2/oob_precision' +str(param_trees[i]) + '.txt', 'w')
        for k in oob_precision_all:
            oob_precision.write(str(k) + "\n")
        oob_precision.close()
#######################Visualization#######################
        for d in class_name:
            filename_t=os.path.join(filePathC,d)
            filename_t=filename_t.replace('\\','/')
            files=os.listdir(filename_t)
            file=os.path.join(filename_t,files[0])
            file=file.replace('\\','/')
            tt=np.loadtxt('C:/Users/lgkgroup/Desktop/a.txt', dtype=float, usecols=1, unpack=True)
            Max = np.max(tt)
            Min= np.min(tt)
            tt=(tt-Min) / (Max-Min)
            xx = max(tt)
            tt = [x/xx for x in tt]
            color_im=num2color(importance_,'Reds')
            plt.figure(figsize=(10,5))
            y_major_locator=MultipleLocator(1)
            ax=plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            plt.ylim(0,1.1) 
            plt.xlim(300,1700) 
            plt.xticks(size=20, family='Times New Roman')
            plt.yticks(size=20, family='Times New Roman')
            plt.bar(feature_name,tt,feature_name[1]-feature_name[0],color=color_im)#cm=colormap
            plt.plot(feature_name,tt,color='k')
#            plt.xlabel('Raman shift',fontsize=15)
#            plt.ylabel('Raman intensity(a.u.)',fontsize=15)
            figpath='C:/Users/lgkgroup/Desktop/test2/'+str(param_trees[i]) +d+'1.png'
            plt.savefig(figpath,dpi=500)
            plt.show()
        figpath='C:/Users/lgkgroup/Desktop/test2/'+'all_Confusion matrix.png'
        plot_confusion_matrix(cm, classes=class_name, figpath=figpath, title='Confusion matrix')
###############################################################
        filename='C:/Users/lgkgroup/Desktop/test2/Improtance(' +str(param_trees[i]) + ').txt'
        file = open(filename, 'w')
        for k, v in zip(list(feature_name),list(importance_)):
            file.write(str(k) + " " + str(v) + "\n")
        file.close()
#            print(metrics.accuracy_score(y_train, rf_clf.predict(X_train)))
#            print(rf_clf.oob_score_)
        print("time_avg:", time_tot / repeat_times)
        print('time_best:', time_best)
        print('time_worst:', time_worst)
        print()
        print('accuracy_avg:', accuracy_tot / repeat_times)
        print('accuracy_best:', accuracy_best)
        print('accuracy_worst:', accuracy_worst)
        print('oob_score_avg: ' , oob_score_tot / repeat_times)
        print('oob_score_best: ', oob_score_best)
        ('oob_score_worst:', oob_score_worst)

        result_f.write("\ntime_avg: " + str(time_tot / repeat_times) + '\n')
        result_f.write('time_best: ' + str(time_best) + '\n')
        result_f.write('time_worst: '+ str(time_worst) + '\n')
        result_f.write('accuracy_avg: ' + str(accuracy_tot / repeat_times) + '\n')
        result_f.write('accuracy_best: '+ str(accuracy_best) + '\n')
        result_f.write('accuracy_worst: '+ str(accuracy_worst) + '\n')
        result_f.write('oob_score_avg: ' + str(oob_score_tot / repeat_times) + '\n')
        result_f.write('oob_score_best: '+ str(oob_score_best) + '\n')
        result_f.write('oob_score_worst: '+ str(oob_score_worst) + '\n')
        result_f.close()

        i = i + 1
####-------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    filePathC = 'C:/Users/lgkgroup/Desktop/test2'
    gci(filePathC, -1)
    a = np.array(a).reshape(data_count,sample_len)
    logo = LeaveOneGroupOut()
    for train_index, test_index in logo.split(a, b, groups=groups):
        for index in train_index:
            if index == iamtrain:
                X_train = a[train_index]
                X_test = a[test_index]
                y_train = b[train_index]
                y_test = b[test_index]
                test_filepath = np.array(data_filepath)[test_index]
                break
#####################CAE################################
    stand='C:/Users/lgkgroup/Desktop/a.txt'
    X_weight=np.loadtxt(stand, dtype=float, usecols=1, unpack=True)
    Max = np.max(X_weight)
    Min= np.min(X_weight)
    X_weight=(X_weight-Min) / (Max-Min)
    X_weight=[x**0.25 for x in X_weight]
    for k in range(X_test.shape[0]):
        X_test[k,]=X_test[k,]*X_weight
        xx = max(X_test[k,])
        X_test[k,] = X_test[k,]/xx
    gcf(X_train, X_test, y_train, y_test)  #train
