import scipy.misc as misc
import matplotlib
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transfers import *
import os
from tqdm import tqdm
#valid_images = cv2.imread('GF2_PMS1__20150212_L1A0000647768-MSS1_label.tif', -1)
#valid_images1=cv2.cvtColor(valid_images,cv2.COLOR_BGR2RGB)
#
#
##灰度化
#gray_image = cv2.cvtColor(valid_images1, cv2.COLOR_BGR2GRAY)
#
##保存灰度图
##matplotlib.image.imsave(r'C:\Users\Administrator\Desktop\python代码\GF2_PMS1__20150212_L1A0000647768-MSS1_label.jpg', gray_image)
##misc.imsave(r'C:\Users\Administrator\Desktop\python代码\GF2_PMS1__20150212_L1A0000647768-MSS1_label1.jpg', gray_image)
#cv2.imwrite(r'C:\Users\Administrator\Desktop\python代码\GF2_PMS1__20150212_L1A0000647768-MSS1_label2.png',gray_image)



# transfer_image(r'GF2_PMS1__20150212_L1A0000647768-MSS1_label.tif',i)




for i in tqdm(os.listdir('G:/WuDaData/rssrai2019_semantic_segmentation/val/GT/')):
    transfer_image('G:/WuDaData/rssrai2019_semantic_segmentation/val/GT/',i)


