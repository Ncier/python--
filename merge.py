# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 09:36:00 2019

@author: A大光
"""

import numpy as np
import cv2
import os
from PIL import Image

"""
输入：图片路径(path+filename)，裁剪所的图片的列的数量、行的数量
输出：无
"""

def merge_picture(merge_path,num_of_cols,num_of_rows):
    filename=file_name(merge_path,".png")
    shape=cv2.imread(filename[0],1).shape    #三通道的影像需把-1改成1
    cols=shape[1]
    rows=shape[0]
    channels=shape[2]
    dst=np.zeros((rows*num_of_rows,cols*num_of_cols,channels),np.uint8)
    for i in range(len(filename)):
        img=cv2.imread(filename[i],1)
        #print(filename[i])
        cols_th=int(filename[i].split("_")[-1].split('.')[0])
        #print(cols_th)
        rows_th=int(filename[i].split("_")[-2])
        #print(rows_th)
        roi=img[0:rows,0:cols,:]
        #print(roi.shape)
        #print((rows_th-1)*rows,(rows_th)*rows,(cols_th-1)*cols,(cols_th)*cols)
        '''
        此处将读取的每一张小图片根据其行数列数填充到dst矩阵中，形成一个大图
        '''
        dst[(rows_th-1)*rows:(rows_th)*rows,(cols_th-1)*cols:(cols_th)*cols,:]=roi
    cv2.imwrite(r"G:\outputa\merge.png",dst)

"""遍历文件夹下某格式图片"""
def file_name(root_path,picturetype):
    filename=[]
    for root,dirs,files in os.walk(root_path):
        for file in files:
            if os.path.splitext(file)[1]==picturetype:
                filename.append(os.path.join(root,file))
    return filename


"""调用合并图片的代码"""
merge_path=r"G:\outputa"   #要合并的小图片所在的文件夹
'''此处的行数列数根据裁剪的情况来变'''
'''拼接会出现图片边缘丢失的情况，原因是裁剪图片的大小不能被原图片大小整除，需要修改裁剪图片的大小'''
num_of_cols=10    #列数
num_of_rows=10     #行数
merge_picture(merge_path,num_of_cols,num_of_rows)