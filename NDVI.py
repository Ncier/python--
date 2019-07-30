# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:28:47 2019

读取包含4个波段的tiff图像，计算NDVI，循环使用不同的阈值，并保存结果图像。

@author: FD
"""

import tifffile as tiff
import cv2
import numpy as np
import matplotlib
import os

# tiff文件
tiff_path = r'E:\data\Dataset-RS\GF1_Model\download\GF1_PMS2_E119.4_N34.9_20170210_L1A0002179101-MSS2.tiff'
# 结果保存路径
save_path = r'E:\data\Dataset-RS\GF1_Model\download\NDVI'
if not os.path.exists(save_path): 
    os.makedirs(save_path)


# 读取图像
img = np.array(tiff.imread(tiff_path))
img = img.astype(float)

# 进行NDVI
nir = img[:, :, 3]
red = img[:, :, 0]
ndvi = (nir-red) / (nir+red)

# 归一化
ndvi_max = ndvi.max()
ndvi_min = ndvi.min()
ndvi_nom = (ndvi+1)/(ndvi_max+1)

# for循环调整阈值，将NDVI结果二值化
print("NDVI---")
for i in range(1, 101):
    # 输出进度条
    if (i%10 ==0) or (i == 1):
        print("\r"+"▇"*int(i/10)+" "+str(i)+"%", end="")
    # 阈值
    a = i/100.0
    # 二值化
    ret, ndvi_res = cv2.threshold(ndvi_nom, a, 255, cv2.THRESH_BINARY)
    # 保存图像
    ndvi_save = save_path + r'\NDVI-' + str(i) + r'.jpg'
    matplotlib.image.imsave(ndvi_save, ndvi_res)
    
print("\nNDVI程序运行结束。")

