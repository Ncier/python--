from mynet import *
import numpy as np
import glob
from keras import backend as K
from skimage import io
import imageio
import cv2
path = r"C:\Users\Administrator\Desktop\2\1\*.tif"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
data_dir = r'D:\论文GPU测试\模型'



#
#
#path1 = r'C:\Users\Administrator\Desktop\2\1'
image_arr = []


image_name_training= glob.glob(os.path.join(path))

for index,item in enumerate(image_name_training):
    

    mat = io.imread(image_name_training[index])
    image = np.transpose(mat,[1,2,0])
    image_arr.append(image)


stride =30

for i_ in range(len(image_arr)):

    item = image_arr[i_]
    h, w, _ = item.shape
    
    
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    padding_img = np.zeros((padding_h, padding_w, 4), dtype=np.uint8)
    padding_img[0:h, 0:w, :] = image[:, :, :]
    padding_img = padding_img.astype("float") / 255.0
    padding_img = img_to_array(padding_img)
    print('src:', padding_img.shape)
    mask_whole = np.zeros((padding_h, padding_w,1), dtype=np.uint8)
    for i in range(padding_h // stride):
        for j in range(padding_w // stride):
            crop = padding_img[ i * stride:i * stride + image_size, j * stride:j * stride + image_size,:4]
            ch, cw,_ = crop.shape
            if ch != 128 or cw != 128:
                print('invalid size!')
                continue

            crop = np.expand_dims(crop, axis=0)
             # print 'crop:',crop.shape
            y_pred_test_img = model.predict(crop, verbose=2)
            
            y_predict = np.array(y_pred_test_img).astype('float32')
            y_predict = np.around(y_predict, 0).astype(np.uint8)
            y_predict *= 255
            y_predict = np.squeeze(y_predict).astype(np.uint8)

             # pred = labelencoder.inverse_transform(pred[0])
             # print (np.unique(pred))
             # pred = pred.reshape((256, 256)).astype(np.uint8)
             # print 'pred:',pred.shape
            mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size,:] = y_pred_test_img[:128, :128,:]
    y_predict = np.squeeze(mask_whole[0:h, 0:w]).astype(np.uint8)

    imx = Image.fromarray(y_predict)
     #

    imx.save("test_outputs/out" + str(i_ + 1) + ".tif")
