"""
data:@20190726
writer：@ Zunqiang Zhao
@ predict the list pictures of recall f1 precision kappa

"""
from PIL import Image
import numpy as np
import sklearn.metrics as metrics
import glob
from skimage import io
import os

data_dir = r'D:\超算文件\qzz\PaperModels\Unet2019411\logs\原来数据训练好的20190715_Unet3'
ground = r"C:\Users\Administrator\Desktop\9\2013.png"


def EvaLuates(modle_dir,ground_dir):
        
    #评价指标函数
    def evaluate(y, y_pre):
        precision = metrics.precision_score(y, y_pre)
        recall    = metrics.recall_score(y, y_pre)
        f1        = metrics.f1_score(y, y_pre)
        kappa     = metrics.cohen_kappa_score(y, y_pre)
        print('precision_score: %f'% (precision))
        print('recall_score: %f'% (recall))
        print('f1_score: %f'% (f1))
        print('kappa: %f'% (kappa))
        print(' ')
    
    def Reads(groundPath, f_name, i,total):
        print('%d / %d' % (i, total))
    
        #读取真值图(255)
        groundTruth = io.imread(groundPath).astype('float32')/255
        width = groundTruth.shape[1]
        groundTruth   = groundTruth.reshape((width*width, 1))
        #读取预测的图(255)
        pred_unet = np.array(Image.open(f_name)).astype('float32')/255
        pred_unet = pred_unet.reshape((width*width, 1))
        #输出文件名
        f_name_ = os.path.basename(os.path.splitext(f_name)[0])
        print(f_name_)
            
        evaluate(groundTruth, pred_unet)


    f_names_jpg = glob.glob(data_dir + '/*.tif')
    f_names_jpg = f_names_jpg[0:]
    
    for i, f_name in enumerate(f_names_jpg):
        Reads(ground, f_name, i+1, total=len(f_names_jpg))

EvaLuates(data_dir,ground)






















#class Metrics(Callback):
#    def on_train_begin(self, logs={}):
#        self.val_f1s = []
#        # self.val_recalls = []
#        # self.val_precisions = []
#
#    def on_epoch_end(self, epoch, logs={}):
#    	val_targ = self.validation_data[1]
#        val_predict = self.model.predict(self.validation_data[0])
#
#        best_threshold = 0
#	    best_f1 = 0
#	    for threshold in [i * 0.01 for i in range(25,45)]:
#	    	y_pred = y_pred=(y_pred > threshold).astype(int)
#	    	# val_recall = recall_score(val_targ, y_pred)
#        	# val_precision = precision_score(val_targ, y_pred)
#	        val_f1 = f1_score(val_targ, val_predict)
#	        if val_f1 > best_f1:
#	            best_threshold = threshold
#	            best_f1 = val_f1
#	            
#        self.val_f1s.append(_val_f1)
#        # self.val_recalls.append(_val_recall)
#        # self.val_precisions.append(_val_precision)
#        print('— val_f1: %f' %(_val_f1))
#        # print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
#        return
#
#
#from keras import backend as K
# 
#def f1(y_true, y_pred):
#    def recall(y_true, y_pred):
#        """Recall metric.
#        Only computes a batch-wise average of recall.
#        Computes the recall, a metric for multi-label classification of
#        how many relevant items are selected.
#        """
#        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#        recall = true_positives / (possible_positives + K.epsilon())
#        return recall
# 
#    def precision(y_true, y_pred):
#        """Precision metric.
#        Only computes a batch-wise average of precision.
#        Computes the precision, a metric for multi-label classification of
#        how many selected items are relevant.
#        """
#        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#        precision = true_positives / (predicted_positives + K.epsilon())
#        return precision
#    precision = precision(y_true, y_pred)
#    recall = recall(y_true, y_pred)
#    return 2*((precision*recall)/(precision+recall+K.epsilon()))
 