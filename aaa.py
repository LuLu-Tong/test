# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:35:05 2021

@author: DongCC
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

CoordinateData=sio.loadmat('coordinates.mat')
BData=sio.loadmat('Bs.mat')
points=CoordinateData['points']
Bs=BData['Bs']

def B(point,coordinate):
    '''
    

    Parameters
    ----------
    point : array
        需要计算磁场的坐标.
    coordinate : list(array)
        决定导线位形的一组坐标（首尾相接）.

    Returns
    -------
    B : array 三维
        point处的磁场.

    '''
    B=np.zeros(3)
    for i in np.arange(len(coordinate)):
        r=point-coordinate[i]
        B = B +  np.cross((coordinate[i-len(coordinate)+1]-coordinate[i]),r) / (np.linalg.norm(r))**3
    return B

def SumDeltaB(coordinate):
    Bs_test=[B(point,coordinate) for point in points]
    DeltaBs=Bs_test-Bs
    return sum([np.linalg.norm(DeltaB) for DeltaB in DeltaBs])

def F(x):
    '''
    需要最优化的函数

    Parameters
    ----------
    x : array
        自变量.

    Returns
    -------
    F : float
        因变量.

    '''
    if len(x)%3 != 0:
        exit(1)
    coordinate=[np.array([x[3*i],x[3*i+1],x[3*i+2]]) for i in np.arange(len(x)//3)]
    return SumDeltaB(coordinate)

def grid(F,x,deltax):
    F_old=F(x)
    gridF=np.zeros(len(x))
    for i in np.arange(len(x)):
        x_=x
        x_[i] = x_[i]+deltax
        gridF[i]=(F(x_)-F_old)/deltax
    return gridF

N=100 #len(coordinate)
steps_max=1000
x=np.zeros(3*N)
deltar_min=1

#steepest descent method 最速下降法
F_old=F(x)
deltar=100.0
for i in np.arange(steps_max):
    gridF=grid(F,x,deltar_min/3)
    gridF_=gridF/np.linalg.norm(gridF)
    F_new=F(x + deltar * gridF_)
    if F_new<F_old:
        x=x+deltar*gridF_
        F_old=F_new
    else:
        deltar=deltar/2
    if deltar<deltar_min:
        print('F=',F_new)
        break
    if i==steps_max-1:
        print('can not find the result in ',steps_max,'steps')

sio.savemat('coordinate_of_i.mat',{'x':x})
# plt.plot()
