# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:26:37 2020

@author: Franx
"""

import numpy as np
import random
import os
from itertools import combinations
import matplotlib.pyplot as plt

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def read(filename):
    file = open(filename,encoding='utf-8')
    data_lines = file.readlines()
    file.close
    orign_keys = []
    orign_values = []
    for data_line in data_lines:
        pair = data_line.split()
        key = float(pair[0])
        value = float(pair[1])
        orign_keys.append(key)
        orign_values.append(value)
    return orign_keys, orign_values

def write(filename, keys, values):
    file = open(filename, 'w')
    for k, v in zip(keys, values):
        file.write(str(k) + " " + str(v) + "\n")
    file.close()

def themoving(filename):
    keys,values=read(filename)
    values_turnleft1=values
    values_turnleft2=values
    values_turnleft4=values
    values_turnleft6=values
    values_turnleft8=values
    del values_turnleft1[0]
    values_turnleft1.insert(len(values_turnleft1)-1,values_turnleft1[len(values_turnleft1)-1])
    del values_turnleft2[0]
    del values_turnleft2[0]
    values_turnleft2.insert(len(values_turnleft2)-1,values_turnleft2[len(values_turnleft2)-1])
    values_turnleft2.insert(len(values_turnleft2)-1,values_turnleft2[len(values_turnleft2)-1])
    del values_turnleft4[0]
    del values_turnleft4[0]
    del values_turnleft4[0]
    del values_turnleft4[0]
    values_turnleft4.insert(len(values_turnleft4)-1,values_turnleft4[len(values_turnleft4)-1])
    values_turnleft4.insert(len(values_turnleft4)-1,values_turnleft4[len(values_turnleft4)-1])
    values_turnleft4.insert(len(values_turnleft4)-1,values_turnleft4[len(values_turnleft4)-1])
    values_turnleft4.insert(len(values_turnleft4)-1,values_turnleft4[len(values_turnleft4)-1])
    del values_turnleft6[0]
    del values_turnleft6[0]
    del values_turnleft6[0]
    del values_turnleft6[0]
    del values_turnleft6[0]
    del values_turnleft6[0]
    values_turnleft6.insert(len(values_turnleft6)-1,values_turnleft6[len(values_turnleft6)-1])
    values_turnleft6.insert(len(values_turnleft6)-1,values_turnleft6[len(values_turnleft6)-1])
    values_turnleft6.insert(len(values_turnleft6)-1,values_turnleft6[len(values_turnleft6)-1])
    values_turnleft6.insert(len(values_turnleft6)-1,values_turnleft6[len(values_turnleft6)-1])
    values_turnleft6.insert(len(values_turnleft6)-1,values_turnleft6[len(values_turnleft6)-1])
    values_turnleft6.insert(len(values_turnleft6)-1,values_turnleft6[len(values_turnleft6)-1])
    del values_turnleft8[0]
    del values_turnleft8[0]
    del values_turnleft8[0]
    del values_turnleft8[0]
    del values_turnleft8[0]
    del values_turnleft8[0]
    del values_turnleft8[0]
    del values_turnleft8[0]
    values_turnleft8.insert(len(values_turnleft8)-1,values_turnleft8[len(values_turnleft8)-1])
    values_turnleft8.insert(len(values_turnleft8)-1,values_turnleft8[len(values_turnleft8)-1])
    values_turnleft8.insert(len(values_turnleft8)-1,values_turnleft8[len(values_turnleft8)-1])
    values_turnleft8.insert(len(values_turnleft8)-1,values_turnleft8[len(values_turnleft8)-1])
    values_turnleft8.insert(len(values_turnleft8)-1,values_turnleft8[len(values_turnleft8)-1])
    values_turnleft8.insert(len(values_turnleft8)-1,values_turnleft8[len(values_turnleft8)-1])
    values_turnleft8.insert(len(values_turnleft8)-1,values_turnleft8[len(values_turnleft8)-1])
    values_turnleft8.insert(len(values_turnleft8)-1,values_turnleft8[len(values_turnleft8)-1])
    return values_turnleft1,values_turnleft2,values_turnleft4,values_turnleft6,values_turnleft8

def addnoise(filename,a):
    keys,values=read(filename)
    values=normalization(values)
    for i in range(len(values)):
        values[i]=values[i]+random.gauss(0,a)
    return values

def line_add(filename1,filename2):#adding spectra(2)
    keys1,values1=read(filename1)
    keys2,values2=read(filename2)
    print(filename1)
    xx = max(values1)
    values1 = [x/xx for x in values1]
    xx = max(values2)
    values2 = [x/xx for x in values2]
    a=random.uniform(0.1,0.9)
    values_mix=[0]*len(values1)
    for i in range(len(values1)):
        values_mix[i]=values1[i]*a+values2[i]*(1-a)
    return keys1,values_mix,a

def line_add3(filename1,filename2,filename3):#adding spectra(3)
    keys1,values1=read(filename1)
    keys2,values2=read(filename2)
    keys3,values3=read(filename3)
    xx = max(values1)
    values1 = [x/xx for x in values1]
    xx = max(values2)
    values2 = [x/xx for x in values2]
    xx = max(values3)
    values3 = [x/xx for x in values3]
    a=random.uniform(0.2,0.4)
    b=random.uniform(0.2,0.4)
    values_mix=[0]*len(values1)
    for i in range(len(values1)):
        values_mix[i]=values1[i]*a+values2[i]*(1-a-b)+values3[i]*b
    return keys1,values_mix,a,b

def line_Mul(filename1):
    keys1,values1=read(filename1)
    a=random.uniform(0.7,0.9)
    values_mix=[0]*len(values1)
    for i in range(len(values1)):
        values_mix[i]=values1[i]*a
    return keys1,values_mix,a

def add_file_direct(folds):
    file=[]
    ext=''
    r_all=0
    filek=os.listdir(folds[0])
#    print(filek)
    file_temp=os.path.join(folds[0],filek[0])
    file_temp=file_temp.replace('\\','/')
    key0,value0=read(file_temp)
    xx = max(value0)
    value0 = [x/xx for x in value0]
    values_all=[0]*len(value0)
    for i in folds:
        files=os.listdir(i)
        file_temp=files[random.randint(0,len(files)-1)]
        file_temp=os.path.join(i,file_temp)
        file_temp=file_temp.replace('\\','/')
        file.append(file_temp)
    for u in file[:len(file)-1]:
        keys,values=read(u)
        xx = max(values)
        values = [x/xx for x in values]
        r=random.uniform(0.1,1/len(file))
        r_all=r_all+r
        name,ext_temp=os.path.split(u)
        ext_temp=ext_temp+str(r)
        values_all=[(values_all[i]+values[i]*r) for i in range(0,len(values_all))]
        ext=ext+ext_temp
    r2=1-r_all
    file_end=file[len(file)-1]
    keys,values=read(file_end)
    xx = max(values)
    values = [x/xx for x in values]
    name,ext_temp=os.path.split(file_end)
    ext_temp=ext_temp+str(r2)
    values_all=[(values_all[i]+values[i]*r) for i in range(0,len(values_all))]
    ext=ext+ext_temp+'.txt'
    return keys,values_all,ext
        
        

def direct_add(top_dir):#Simulation data generation by adding categories in order
    dirs=os.listdir(top_dir)
    dirs.sort()
    print(dirs)
    r=4000#numbers
    dirs_detail=[]
    for m in range(len(dirs)):
        file_name=os.path.join(top_dir,dirs[m])
        dirs_detail.append(file_name)
    for n in range(2,len(dirs_detail)+1):
        dirs_temp=dirs_detail[0:n]
        n=str(n)*2
        k=os.path.join(top_dir,n+'mix')
        isExists=os.path.exists(k)
        if not isExists:
            os.makedirs(k) 
        for q in range(r):
            keys,values_all_temp,ext=add_file_direct(dirs_temp)
            path=os.path.join(k,ext)
            write(path,keys,values_all_temp)
        

def main(top_dir):#List all the possibilities of the mixture, select the folder and randomly select the file for mixing
    dirs=os.listdir(top_dir)
    dirs.sort()
    print(dirs)
    for m,n in combinations(dirs,2):
        k=os.path.join(top_dir,m+'+'+n)
        k=k.replace('\\','/')
        isExists=os.path.exists(k)
        if not isExists:
            os.makedirs(k) 
        dir1=os.path.join(top_dir,m)
        dir1=dir1.replace('\\','/')
        dir2=os.path.join(top_dir,n)
        dir2=dir2.replace('\\','/')
        files1=os.listdir(dir1)
        files2=os.listdir(dir2)
        r=4000#numbers
        for i in range(r):
            print(files2)
            a=random.randint(0,len(files1)-1)
            b=random.randint(0,len(files2)-1)
            file1=files1[a]
            file2=files2[b]
            file1_1=os.path.join(dir1,file1)
            file2_2=os.path.join(dir2,file2)
            keys,values_mix,o=line_add(file1_1,file2_2)
            new_name=file1+str(o)+'mix'+str(1-o)+file2
            path=os.path.join(k,new_name)
            path=path.replace('\\','/')
            write(path,keys,values_mix)
#    for m,n,l in combinations(dirs,3):
#        k=os.path.join(top_dir,m+'+'+n+'+'+l)
#        k=k.replace('\\','/')
#        isExists=os.path.exists(k)
#        if not isExists:
#            os.makedirs(k) 
#        dir1=os.path.join(top_dir,m)
#        dir1=dir1.replace('\\','/')
#        dir2=os.path.join(top_dir,n)
#        dir2=dir2.replace('\\','/')
#        dir3=os.path.join(top_dir,l)
#        dir3=dir3.replace('\\','/')
#        files1=os.listdir(dir1)
#        files2=os.listdir(dir2)
#        files3=os.listdir(dir3)
#        r=200
#        for i in range(r):
#            a=random.randint(0,len(files1)-1)
#            b=random.randint(0,len(files2)-1)
#            c=random.randint(0,len(files3)-1)
#            file1=files1[a]
#            file2=files2[b]
#            file3=files3[c]
#            file1_1=os.path.join(dir1,file1)
#            file2_2=os.path.join(dir2,file2)
#            file3_3=os.path.join(dir3,file3)
#            keys,values_mix,o,p=line_add3(file1_1,file2_2,file3_3)
#            new_name=file1+str(o)+'mix'+file2+str(p)+'mix'+str(1-o-p)+file3
#            path=os.path.join(k,new_name)
#            path=path.replace('\\','/')
#            write(path,keys,values_mix)

def main1(top_dir):#Left or right moving
    dirs=os.listdir(top_dir)
    dirs.sort()
    print(dirs)
    for i in dirs:
        dir_=os.path.join(top_dir+'/'+i)
        files=os.listdir(dir_)
        for k in files:
            file=os.path.join(dir_+'/'+k)
            keys,values=read(file)
            values_turnleft1,values_turnleft2,values_turnleft4,values_turnleft6,values_turnleft8=themoving(file)
            write(os.path.join(file + 'left1.txt'), keys, values_turnleft1)
            write(os.path.join(file + 'left2.txt'), keys, values_turnleft2)
            write(os.path.join(file + 'left4.txt'), keys, values_turnleft4)
            write(os.path.join(file + 'left6.txt'), keys, values_turnleft6)
            write(os.path.join(file + 'left8.txt'), keys, values_turnleft8)
        
def main2(top_dir):#adding noise
    dirs=os.listdir(top_dir)
    dirs.sort()
    tot=0
    print(dirs)
    for i in dirs:
        dir_=os.path.join(top_dir+'/'+i)
        files=os.listdir(dir_)
        for m in range(1):
            for r in range(len(files)):
                k=files[r]
                file=os.path.join(dir_+'/'+k)
                keys,values=read(file)
                values=normalization(values)
                u=random.uniform(0,0.15)
                values_addnoise_low=addnoise(file,u)
                write(os.path.join(file+str(tot)+'.txt'), keys, values_addnoise_low)
                tot=tot+1
            
#————————test————————#
if __name__ == "__main__":
        main1('C:/Users/lgkgroup/Desktop/test5')

