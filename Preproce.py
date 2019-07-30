import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt



def make_train_txt():
    raw_img_path = r'G:\绿潮数据集\绿潮数据集2\80x80\images\training\*.png'#G:\绿潮数据集\绿潮数据集2\预测2000x2000\原图
    imgs_path = glob.glob(raw_img_path)

    txt_path = r'G:\绿潮数据集\绿潮数据集2\80x80\train.txt'
    txt = open(txt_path, 'w')
    for img_path in imgs_path:
        print(img_path)
        # raw depth instance semantic
        data = img_path + '\n'
        txt.write(data)

def make_train_txt2():
    raw_img_path = r'G:\绿潮数据集\绿潮数据集2\80x80\annotations\training\*.png'#G:\绿潮数据集\绿潮数据集2\预测2000x2000\原图
    imgs_path = glob.glob(raw_img_path)

    txt_path = r'G:\绿潮数据集\绿潮数据集2\80x80\lable.txt'
    txt = open(txt_path, 'w')
    for img_path in imgs_path:
        print(img_path)
        # raw depth instance semantic
        data = img_path + '\n'
        txt.write(data)


make_train_txt()
make_train_txt2()