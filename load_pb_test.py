import tensorflow as tf
import numpy as np
import cv2
import scipy.io   as scio

from skimage import transform,data 

"""-----------------------------------------------定义识别函数-----------------------------------------"""
def recognize(jpg_path, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
 
        # 打开.pb模型
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
            print("tensors111111111111111111111:",tensors)
 
        # 在一个session中去run一个前向
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
 
            op = sess.graph.get_operations()
 
            # 打印图中有的操作
            for i,m in enumerate(op):
                print('op{}:'.format(i),m.values())
            
            input_x = sess.graph.get_tensor_by_name("input_1:0")  # 具体名称看上一段代码的input.name
            
           
            
            print("input_X:",input_x)
 
            out_softmax = sess.graph.get_tensor_by_name("conv2d_39/Sigmoid:0")  # 具体名称看上一段代码的output.name
            print("Output:",out_softmax)
 
            # 读入图片
                
#            img =  scio.loadmat(jpg_path)  
            
            
            img =  (scio.loadmat(jpg_path)['data'])
            img = img[np.newaxis, :] 
            #img = cv2.imread(jpg_path, -1)
#            img=cv2.resize(img,(24,24))
            img=img.astype(np.float32)
            img=img/255;
#            img = img[np.newaxis, :]
            
          
           
            image = tf.placeholder(tf.float32, shape=[1, 600, 600,2], name="input_image")
            # img=np.reshape(img,(1,28,28,1))
            print("img data type:",img.dtype)
 
#            # 显示图片内容
#            for row in range(24):
#                for col in range(24):
#                    if col!=23:
#                        print(img[row][col],' ',end='')
#                    else:
#                        print(img[row][col])
 
           
    
            img_out_softmax = sess.run(out_softmax,
                                       feed_dict={input_x:img})
 
            print("img_out_softmax:", img_out_softmax)
            for i,prob in enumerate(img_out_softmax[0]):
                print('class {} prob:{}'.format(i,prob))
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print("Final class if:",prediction_labels)
            print("prob of label:",img_out_softmax[0,prediction_labels])
 
 
pb_path = './unet.pb'
img = 'C:/Users/Administrator/Desktop/python代码/picture/mat.mat'
recognize(img, pb_path)






#
#Using TensorFlow backend.
#input is : input_1:0
#output is: conv2d_39/Sigmoid:0
#INFO:tensorflow:Froze 325 variables.
#INFO:tensorflow:Converted 325 variables to const ops.








