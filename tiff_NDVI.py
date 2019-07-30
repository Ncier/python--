"""
# -*- coding: utf-8 -*-
归一化植被指数：NDVI
增强型植被指数：EVI
比值植被指数：RVI
差值植被指数：DVI
浮动藻类知书：FAI
表面藻华指数SABI
GOCI浮动绿藻指数：IGAG
"""
import keras
import tifffile as tiff
import cv2
import numpy as np
import matplotlib
import os
import imageio
from decimal import *

import scipy.io   as scio
# tiff文件
tiff_path = r"C:\Users\Administrator\Desktop\画的比较好的\2016标注\20160624.tif"
# 结果保存路径
save_path = r'C:\Users\Administrator\Desktop\画的比较好的\NDVI\1'
if not os.path.exists(save_path): 
    os.makedirs(save_path)


# 读取图像 
img = np.array(tiff.imread(tiff_path)).transpose(1,2,0)
#img = scio.loadmat(tiff_path)
#img =img['data']  
#
#a ,b =cv2.split(img)
#
#test_img = img[:, -200:, :]








#q ,w, e=img.shape
#c=img.ravel()
#c1 =np.array(c.reshape((e,q*w)))
#c2 =np.array(np.transpose(c1))
#
#
#
#c3=np.dot(c2,c1)
#
#c4 = keras.backend.softmax(c3)
#
#c5 = np.dot(c3,c2)
#
#c6 = c5.reshape(e,w,q)
#
#
#
#
#
#image = c6
#image=(image-np.min(image))/(np.max(image)-np.min(image))
#
#
#
#
#
#print("d=",c4)








# 进行NDVI
nir = img[:, :, 1]
red = img[:, :, 0]
#归一化植被指数
ndvi = (nir-red) / (nir+red)

##增强型植被指数
#evi = ((nir-red)/(nir+6*red-7.5*red+1))*2.5

##比值植被指数
#rvi = nir/red

##差值植被指数
#dvi = nir-red

##浮动藻类指数
#fai = nir-red-





## 归一化
#ndvi_max = ndvi.max()
#ndvi_min = ndvi.min()
#ndvi_nom = (ndvi+1)/(ndvi_max+1)

# for循环调整阈值，将NDVI结果二值化
print("NDVI---")
a=-0.5
for i in range(1, 200):
    # 输出进度条
    if (i%10 ==0) or (i == 1):
        print("\r"+"▇"*int(i/10)+" "+str(i)+"%", end="")
    # 阈值
   
    # 二值化
    ret, ndvi_res = cv2.threshold(ndvi, round(a,2), 255, cv2.THRESH_BINARY)
    
#    ndvi_res = ndvi_res.astype("uint8")
    
    # 保存图像
    ndvi_save = save_path + r'\NDVI=' + str(round(a,2)) + r'.tif'
    
    imageio.imwrite(ndvi_save, ndvi_res)  
    a = 0.01+a
    
    a=round(a,2)
    
    
print("\nNDVI程序运行结束。")

