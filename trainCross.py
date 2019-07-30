# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:45:39 2019

@author: Administrator
"""

from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.callbacks import ModelCheckpoint
from Models import model_unet, model_psp, mdoel_segnet, model_Deeplabv3P, model_ourNet

# 超参数
K = 10  # 交叉验证
IMAGE_SIZE = 64
batch_size = 16
epochs = 10
classes = 9
LR = 1e-4
input_sizes=(IMAGE_SIZE,IMAGE_SIZE,4)

#===================================================================================================
'''
    加载模型信息
'''
model_name_list = ['UNet', 'PSPNet', 'SegNet', 'Deeplabv3P', 'OurNet']
model_name = model_name_list[0]

# 导入模型
if model_name == model_name_list[0]:
    model = model_unet.unet(input_size=input_sizes, num_class=classes, model_summary=True)
    print('==========================Start {}!======================================='.format(model_name))
elif model_name == model_name_list[1]:
    model = model_psp.psp_net(input_size=input_sizes, num_class=classes, LR=LR, model_summary=True)
    print('==========================Start {}!======================================='.format(model_name))
elif model_name == model_name_list[2]:
    model = mdoel_segnet.SegNet(input_size=input_sizes, num_class=classes, model_summary=True)
    print('==========================Start {}!======================================='.format(model_name))
elif model_name == model_name_list[3]:
    model = model_Deeplabv3P.Deeplabv3(input_size=input_sizes, classes=classes, LR=LR, model_summary=True)
    print('==========================Start {}!======================================='.format(model_name))
elif model_name == model_name_list[4]:
    model = model_ourNet.ourNet(input_size=input_sizes, num_class=classes, model_summary=True)
    print('==========================Start {}!======================================='.format(model_name))
else:
    print('No Model!')
#===================================================================================================
'''
    加载数据集
'''

# 训练集
images = np.load(os.path.join(os.getcwd(), 'Train_Npy', 'imagesDataset_.npy'))
lables = np.load(os.path.join(os.getcwd(), 'Train_Npy', 'labelsDataset_.npy'))
lables = lables.astype('float32')

#===================================================================================================

'''
    训练模型
'''
def model_set(model_name, k):
    
    save_dir = os.path.join(os.getcwd(), 'Saved_models', model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 模型保存路径
    save_model_path = os.path.join(save_dir, model_name+'-K:'+str(k)+'-{epoch:02d}-{val_loss:.4f}.hdf5')
    model_checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss',verbose=1, save_best_only=True, mode='auto', period=1, save_weights_only=True)
    
    # 日志信息
#    weights_save_path = os.path.join(save_dir, model_name+'_weights')
#    tensorboard = TensorBoard(log_dir=weights_save_path, histogram_freq=1)

    # 早停
#    EarlyStoppings = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    
    # 学习率减小
#    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    #callback_lists = [model_checkpoint, EarlyStopping, tensorboard, reduce_lr]
    callback_lists = [model_checkpoint]

    return callback_lists



# K 折交叉验证
skf = StratifiedKFold(n_splits=K)

# 设置交叉验证时必须的参数 y
y = len(lables)
y = np.zeros(y)

n = 0
for train, valid in skf.split(images, y):
    
    n+=1
    print('------------------------------K = {:<2d}({})------------------------------'.format(n, K))

    # 模型设置
    callback_lists = model_set(model_name, n)
    
    train_img = images[train, :, :, :]
    train_lab = lables[train, :, :, :]
    
    valid_img = images[valid, :, :, :]
    valid_lab = lables[valid, :, :, :]

    # 训练
    model.fit(train_img,
              train_lab,
              validation_data=(valid_img, valid_lab),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=callback_lists)