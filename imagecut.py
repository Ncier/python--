'''
writer：@ Zunqiang Zhao
实现功能：实现批量输入.png;.jpg图像，输出到指定路径下（适合裁切真值图）
fun:CutGroundTruth（）裁切真值图
fun：CutImage（）裁切modis图像
此代码没有加数据扩充功能（<---- >）
'''
import cv2
from PIL import Image
import glob
import os
import numpy as np
import scipy.io   as scio
from skimage import io
from skimage import  img_as_float32
import matplotlib.pyplot as plt



#mat = scio.loadmat(r"C:\Users\Administrator\Desktop\2\2\20130620_cut_1.mat")
#maimage = mat['data']  np.max   



def CutGroundTruth(vx,vy,inpath,outpath,cutsubs=True):
    #遍历文件夹下所有的图片
    imgs_path = glob.glob(inpath + "*.png")
    a = 0
    mask_arr = []
    print("裁切出图像大小为%s*%s"%(vx,vy))
    for img_path in imgs_path:
        
        im = Image.open(img_path)         
        m= os.path.split(img_path)[-1].split(".")[0]
        
        #偏移量
        dx = vx
        dy = vy
        n = 1
    
        #左上角切割
        x1 = 0
        y1 = 0
        x2 = vx
        y2 = vy
    
        #纵向
        while x2 <= im.size[0]:
            #横向切
            while y2 <= im.size[1]:
                name3 = outpath + m +"_cut_"+ str(n) + ".png"
                #print n,x1,y1,x2,y2 roi=img[80:180,100:200,:]
                im2 = im.crop((y1, x1, y2, x2))
                
                #im2=(im2-np.min(im2))/(np.max(im2)-np.max(im2))
                
                im21 = img_as_float32(im2)
                im22 = im21[:, :,np.newaxis] 
                if cutsubs:
                    
                    im2.save(name3)
                
                mask_arr.append(im22)                
                mask_arr1 = np.array(mask_arr)  
                y1 = y1 + dy
                y2 = y1 + vy
                n = n + 1
            x1 = x1 + dx
            x2 = x1 + vx
            y1 = 0
            y2 = vy
            
        print ("图片"+ m +"大小为"+str(im.size[0])+"x"+str(im.size[1])+",切割得到的子图片数为：",n-1) 
        a = (n-1)+a
    print ("切割的图片总数为：",a)
    return mask_arr,mask_arr1
             
def CutImage(vx,vy,inpath,outpath,cutsubs=True):
    
     #遍历文件夹下所有的图片
    imgs_path = glob.glob(inpath + "*.tif")
    a = 0
    image_arr = []
    print("裁切出图像大小为%s*%s"%(vx,vy))
    for img_path in imgs_path:
        
        im = io.imread(img_path)  
        im1 = im.transpose(1,2,0)
    
        
        m= os.path.split(img_path)[-1].split(".")[0]
        
        #偏移量
        dx = vx
        dy = vy
        n = 1
    
        #左上角切割
        x1 = 0
        y1 = 0
        x2 = vx
        y2 = vy
    
        #纵向
        while x2 <= im1.shape[0]:
            #横向切
            while y2 <= im1.shape[1]:
                name3 = outpath + m +"_cut_"+ str(n) + ".mat"
                #print n,x1,y1,x2,y2
              #  im2 = im1.crop((y1, x1, y2, x2))
                im2 = im1[x1:x2,y1:y2,:]  
                if cutsubs:
                            
                    scio.savemat(name3,{"data":im2})
                
                image_arr.append(im2)                
                image_arr1 = np.array(image_arr)                
                y1 = y1 + dy
                y2 = y1 + vy
                n = n + 1
            x1 = x1 + dx
            x2 = x1 + vx
            y1 = 0
            y2 = vy
            
        print ("图片"+ m +"大小为"+str(im1.shape[0])+"x"+str(im1.shape[1])+",切割得到的子图片数为：",n-1) 
        a = (n-1)+a
    print ("切割的图片总数为：",a)             
    return image_arr,image_arr1

#CutImage CutGroundTruth
# 训练集   images_training  annotations_training     验证集   images_validation  annotations_validation  
def out(x,y,input1,input2,output,cutsubs=True):#True False
    
     if cutsubs:
         
             image_arr,training_images = CutImage(x,y,input1,output,cutsubs=True)
             image_arr,training_annotations = CutGroundTruth(x,y,input1,path3,cutsubs=True)
             
             image_arr,validation_images = CutImage(x,y,input2,output,cutsubs=True)
             image_arr,validation_annotations = CutGroundTruth(x,y,input2,path3,cutsubs=True)
             
     image_arr,training_images = CutImage(x,y,input1,output,cutsubs=False)
     image_arr,training_annotations = CutGroundTruth(x,y,input1,path3,cutsubs=False)
     
     image_arr,validation_images = CutImage(x,y,input2,output,cutsubs=False)
     image_arr,validation_annotations = CutGroundTruth(x,y,input2,path3,cutsubs=False)
     
     
     np.save(output+"images_training.npy",training_images)
     np.save(output+"annotations_training.npy",training_annotations)
     
     np.save(output+"images_validation.npy",validation_images)
     np.save(output+"annotations_validation.npy",validation_annotations)
          
if __name__=="__main__":

   #输入，输出路径 C:\Users\Administrator\Desktop\2\2
   path1= r"C:/Users/Administrator/Desktop/2/"#训练集路径
   path2= r"C:/Users/Administrator/Desktop/2/"#验证集路径   
   path3 = r"C:/Users/Administrator/Desktop/2/image3/"#输出路径
    
   out(24,24,path1,path2,path3)
    
    
   
   
   
   
   
   
   
   
   
