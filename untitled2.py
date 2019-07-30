#-*-coding:utf_*_
import numpy as np
import cv2
import os
"""
输入：图片路径(path+filename)，裁剪获得小图片的列数、行数（也即宽、高）
"""
def clip_one_picture(path,filename,cols,rows):
    img=cv2.imread(r"C:\Users\Administrator\Desktop\u=2883905864,3074928606&fm=26&gp=0.jpg",-1)##读取彩色图像，图像的透明度(alpha通道)被忽略，默认参数;灰度图像;读取原始图像，包括alpha通道;可以用1，0，-1来表示
    sum_rows=img.shape[0]   #高度
    sum_cols=img.shape[1]    #宽度
    save_path=path+"\\crop{0}_{1}\\".format(cols,rows)  #保存的路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("裁剪所得{0}列图片，{1}行图片.".format(int(sum_cols/cols),int(sum_rows/rows)))

    for i in range(int(sum_cols/cols)):
        for j in range(int(sum_rows/rows)):
            cv2.imwrite(save_path+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+os.path.splitext(filename)[1],img[j*rows:(j+1)*rows,i*cols:(i+1)*cols,:])
            #print(path+"\crop\\"+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+os.path.splitext(filename)[1])
    print("裁剪完成，得到{0}张图片.".format(int(sum_cols/cols)*int(sum_rows/rows)))
    print("裁剪所得图片的存放地址为：{0}".format(save_path))


"""调用裁剪函数示例"""
path='C:\\Users\\Administrator\\Desktop\\尹蓓542\\投影\\'   #要裁剪的图片所在的文件夹
filename='GF1_WFV1_E119.7_N34.7_20170310_L1A0002230542.tiff'    #要裁剪的图片名
cols=32        #小图片的宽度（列数）
rows=32        #小图片的高度（行数）
clip_one_picture(path,filename,32,32)
