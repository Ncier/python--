from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as skimageio
import skimage.transform as trans
import cv2
import scipy
import datetime
import scipy.io   as scio

from PIL import Image, ImageFont
from skimage import io


import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from skimage import  img_as_float32
import imageio




#masks =  cv2.imdecode(np.fromfile(r"C:\Users\Administrator\Desktop\2\20170608.png",dtype=np.uint8),cv2.IMREAD_LOAD_GDAL)
#masks = img_as_float32(masks)
#img = np.load(r"C:\Users\Administrator\Desktop\原来\annotations_training.npy")
#img1= np.load(r"C:\Users\Administrator\Desktop\2\annotations_training.npy")
#
#
#img原来_1 = img[8,:,:,:]
#img原来_1712 = img[1711,:,:,:]
#
#img1_1= img1[8,:,:,:]
#img1_1712 = img1[9,:,:,:]










#image_arr = []
#mask_arr = []
#path=r'C:\Users\Administrator\Desktop\2\20180608.tif'#测试两个波段tif
#path1=r'C:\Users\Administrator\Desktop\2\20180608.mat'#测试两个波段mat
#path2=r'C:\Users\Administrator\Desktop\2\20180608.png'#测试一个波段mat
#
#mat = scio.loadmat(path1)['data'] 
#mat1 = img_as_float32(mat)
#
#png = cv2.imread(path2, -1)
#png1 = img_as_float32(png)
#png2 = png1[:, :,np.newaxis]   
#                                      
##data = cv2.imread(path, -1)
#tif_1= skimageio.imread(path)
#tif_1_1= img_as_float32(tif_1)
#
##data_2 = scipy.misc.imread(path)
#tif_3 = imageio.imread(path)
#tif_4 = io.imread(path)
#
#
#
#image_arr.append(mat)
#mask_arr.append(tif_1)
#mask_arr.append(tif_3)
#
#image_arr1 = np.array(image_arr)
#mask_arr1 = np.array(mask_arr)

imagedir =r"C:\Users\Administrator\Desktop\画的比较好的\image"
maskdir =r"C:\Users\Administrator\Desktop\画的比较好的\validation"
outputdir =r"C:\Users\Administrator\Desktop\画的比较好的/"

test_image=r"C:\Users\Administrator\Desktop\3"
test_mask =r"C:\Users\Administrator\Desktop\3"


image_training=r"D:\互联网数据\黄河\image"#validation training
mask_training=r"D:\互联网数据\黄河\GT"#validation training
image_validation=r"D:\互联网数据\黄河\image2"#validation training
mask_validation=r"D:\互联网数据\黄河\GT2"#validation training

#image_training=r"F:\深度学习项目\绿潮数据集\模板2\images\training"#validation training
#mask_training=r"F:\深度学习项目\绿潮数据集\模板2\annotations\training"#validation training
#image_validation=r"F:\深度学习项目\绿潮数据集\模板2\images\validation"#validation training
#mask_validation=r"F:\深度学习项目\绿潮数据集\模板2\annotations\validation"#validation training



##mat-mat
def geneTrainNpy(image_path,mask_path):
    image_name_training= glob.glob(os.path.join(image_path,"*.mat"))
    mask_name_training  = glob.glob(os.path.join(mask_path,"*.mat"))
#    image_name_validation= glob.glob(os.path.join(image_ptraining,"*.mat"))
#    mask_name_validation  = glob.glob(os.path.join(mask_training,"*.mat"))
    
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_training):
        #img = cv2.imread(item, -1)
        img = scio.loadmat(item)               
        image=(img['data'])
        
        image=(image-np.min(image))/(np.max(image)-np.min(image))
        #image = img_as_float32(image)
        
        #img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        mask = scio.loadmat(mask_name_training[index])
        masks=(mask['data'])
        masks = img_as_float32(masks)
        #mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        masks = masks[:, :,np.newaxis]
       
        image_arr.append(image)
        mask_arr.append(masks)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

##mat-png
def geneTrainNpy2(test_image,test_mask):
    image_name_training= glob.glob(os.path.join(test_image,"*.mat"))
    mask_name_training  = glob.glob(os.path.join(test_mask,"*.png"))
#    image_name_validation= glob.glob(os.path.join(image_ptraining,"*.mat"))
#    mask_name_validation  = glob.glob(os.path.join(mask_training,"*.mat"))
    
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_training):
        #img = cv2.imread(item, -1)
        img = scio.loadmat(item)               
        image=(img['data'])
        
             
                        
#        #NDVI 方法一
#        red=image[:,:,0] 
#        nir=image [:,:,1]
#        ndvi=(nir-red) / (nir+red)
#        ndvi1=(ndvi-np.min(ndvi))/(np.max(ndvi)-np.min(ndvi))
#        
#        ret, ndvi_res = cv2.threshold(ndvi, -0.1, 1, cv2.THRESH_BINARY)
        
        
        image=(image-np.min(image))/(np.max(image)-np.min(image))
        
#        image1,image2 =cv2.split(image)
#        
#        image = cv2.merge([image1,image2,ndvi_res]) #合并
        
        
        
        
#        #(1,2,1)假彩色合成 方法二
#        image1,image2 =cv2.split(image)
#        image = cv2.merge([image1,image2,image1]) #合并
#        image=(image-np.min(image))/(np.max(image)-np.min(image))
#        
        
        
       # image = img_as_float32(image)
        
#        mask = scio.loadmat(massk_name_training[index])
#        masks=(mask['data'])
#        masks = img_as_float32(masks)       
#        masks = masks[:, :,np.newaxis]
       
        masks =  cv2.imdecode(np.fromfile(mask_name_training[index],dtype=np.uint8),cv2.IMREAD_LOAD_GDAL)#255
        masks = img_as_float32(masks)#255 -> 1
        masks = masks[:, :,np.newaxis]   
                                      
        image_arr.append(image)
        mask_arr.append(masks)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr





##mat-mat
#image_trainings,mask_trainings=geneTrainNpy(image_training,mask_training)
#np.save(r"C:\Users\Administrator\Desktop\黄河\images_training.npy",image_trainings)
#np.save(r"C:\Users\Administrator\Desktop\黄河\annotations_training.npy",mask_trainings)
#
#
#image_validations,mask_validations=geneTrainNpy(image_validation,mask_validation)
#np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\GreenDate\images_validation.npy",image_validations)
#np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\GreenDate\annotations_validation.npy",mask_validations)


#mat-png
image_trainings,mask_trainings=geneTrainNpy2(imagedir,imagedir)
np.save(outputdir+"images_training.npy",image_trainings)
np.save(outputdir+"annotations_training.npy",mask_trainings)


image_validations,mask_validations=geneTrainNpy2(maskdir,maskdir)
np.save(outputdir+"images_validation.npy",image_validations)
np.save(outputdir+"annotations_validation.npy",mask_validations)


#
###test-mat-png
#test_images,test_masks=geneTrainNpy2(test_image,test_mask)
#np.save(r"C:\Users\Administrator\Desktop\3\test_images.npy",test_images)
#np.save(r"C:\Users\Administrator\Desktop\3\test_masks.npy",test_masks)




























