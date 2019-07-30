##!/usr/bin/python
## coding:utf8
###-------keras模型保存为tensorflow的二进制模型-----------
#import sys
#from mynet import *
#from keras.models import load_model
#import tensorflow as tf
#import os
#import os.path as osp
#from keras import backend as K
#from tensorflow.python.framework.graph_util import convert_variables_to_constants
#
#def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#    # 将会话状态冻结为已删除的计算图,创建一个新的计算图,其中变量节点由在会话中获取其当前值的常量替换.
#    # session要冻结的TensorFlow会话,keep_var_names不应冻结的变量名列表,或者无冻结图中的所有变量
#    # output_names相关图输出的名称,clear_devices从图中删除设备以获得更好的可移植性
#    graph = session.graph
#    with graph.as_default():
#        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#        output_names = output_names or []
#        output_names += [v.op.name for v in tf.global_variables()]
#        input_graph_def = graph.as_graph_def()
#        # 从图中删除设备以获得更好的可移植性
#        if clear_devices:
#            for node in input_graph_def.node:
#                node.device = ""
#        # 用相同值的常量替换图中的所有变量
#        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
#        return frozen_graph
##
##output_fld = sys.path[0] + '/C:/Users/Administrator/Desktop/tests1/data/'
##if not os.path.isdir(output_fld):
##    os.mkdir(output_fld)
##weight_file_path = osp.join(sys.path[0], 'tmp/first_try.hdf5')
#output_fld = r"C:\Users\Administrator\Desktop\tests1\data"
#weight_file_path  = r"C:\Users\Administrator\Desktop\tests1\data\ert.hdf5"
#K.set_learning_phase(0)
#
#
#model1 = unet3(2, (32,32,4), 1, 1, 0.0001, Falg_summary=False, Falg_plot_model=False)
#
#net_model = model1.load_weights("ert.h5")
#
##print('input is :', net_model.pool1.name)
##print ('output is:', net_model.pool2.name)
#
## 获得当前图
#sess = K.get_session()
## 冻结图
#frozen_graph = freeze_session(sess, output_names=[net_model.output.op.name])
#
#from tensorflow.python.framework import graph_io
#graph_io.write_graph(frozen_graph, output_fld, 'new_tensor_model.pb', as_text=False)
#print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, 'new_tensor_model.pb'))
#print (K.get_uid())
#
#
#

#
#import cv2
#from mynet import *
#import numpy as np
#from keras.models import load_model
#model1 = unet3(2, (32,32,4), 1, 1, 0.0001, Falg_summary=False, Falg_plot_model=False)
#model = model1.load_weights('ert.h5')  #选取自己的.h模型名称
#
#image = cv2.imread('6_b.png')
#img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # RGB图像转为gray
#
# #需要用reshape定义出例子的个数，图片的 通道数，图片的长与宽。具体的参加keras文档
#img = (img.reshape(1, 1, 28, 28)).astype('int32')/255 
#predict = model.predict_classes(img)
#print ('识别为：')
#print (predict)
#
#cv2.imshow("Image1", image)
#cv2.waitKey(0)

import keras
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_io
from mynet import * 
from keras.optimizers import SGD, rmsprop, Adam





def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph
 
 
"""----------------------------------配置路径-----------------------------------"""
epochs=100
#h5_model_path='ert.hdf5'.format(epochs)
output_path=r"C:\Users\Administrator\Desktop\python代码"
pb_model_name='unet.pb'.format(epochs)
 
 
"""----------------------------------导入keras模型------------------------------"""
K.set_learning_phase(0)
net_model = keras.models.load_model(r"C:\Users\Administrator\Desktop\python代码\asd.hdf5".format(epochs))
 
print('input is :', net_model.input.name)
print ('output is:', net_model.output.name)
 
"""----------------------------------保存为.pb格式------------------------------"""
sess = K.get_session()
frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])
graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)

















