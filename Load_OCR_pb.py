import tensorflow as tf
import  numpy as np
import PIL.Image as Image
import cv2
# from skimage import transform
W = 64
H = 64

#可以读取带中文路径的图
def cv_imread(file_path,type=0):
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    # print(file_path)
    # print(cv_img.shape)
    # print(len(cv_img.shape))
    if(type==0):
        if(len(cv_img.shape)==3):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv_img

#叠加两张图片，输入皆是黑白图，img1是底层图片，img2是上层图片，返回叠加后的图片
def ImageOverlay(img1,img2):
    # 把logo放在左上角，所以我们只关心这一块区域
    h = img1.shape[0]
    w = img1.shape[1]
    rows = img2.shape[0]
    cols = img2.shape[1]
    roi = img1[int((h - rows) / 2):rows + int((h - rows) / 2), int((w - cols) / 2):cols + int((w - cols) / 2)]
    # 创建掩膜
    img2gray = img2.copy()
    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask)
    # 保留除logo外的背景
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    dst = cv2.add(img1_bg, img2)  # 进行融合
    img1[int((h - rows) / 2):rows + int((h - rows) / 2),int((w - cols) / 2):cols + int((w - cols) / 2)] = dst  # 融合后放在原图上
    return img1

# 处理白边
#找到上下左右的白边位置
#剪切掉白边
#二值化
#将图像放到64*64的白底图像中心
def HandWhiteEdges(img):
    ret, thresh1 = cv2.threshold(img, 249, 255, cv2.THRESH_BINARY)
    # OpenCV定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 膨胀图像
    thresh1 = cv2.dilate(thresh1, kernel)
    row= img.shape[0]
    col = img.shape[1]
    tempr0 = 0    #横上
    tempr1 = 0    #横下
    tempc0 = 0    #竖左
    tempc1 = 0    #竖右
    # 765 是255+255+255,如果是黑色背景就是0+0+0，彩色的背景，将765替换成其他颜色的RGB之和，这个会有一点问题，因为三个和相同但颜色不一定同
    for r in range(0, row):
        if thresh1.sum(axis=1)[r] != 255 * col:
            tempr0 = r
            break

    for r in range(row - 1, 0, -1):
        if thresh1.sum(axis=1)[r] != 255 * col:
            tempr1 = r
            break

    for c in range(0, col):
        if thresh1.sum(axis=0)[c] != 255 * row:
            tempc0 = c
            break

    for c in range(col - 1, 0, -1):
        if thresh1.sum(axis=0)[c] != 255 * row:
            tempc1 = c
            break

    # 创建全白图片
    imageTemp = np.zeros((64, 64, 3), dtype=np.uint8)
    imageTemp = cv2.cvtColor(imageTemp, cv2.COLOR_BGR2GRAY)
    imageTemp.fill(255)

    if(tempr1-tempr0==0 or tempc1-tempc0==0):   #空图
        return imageTemp    #返回全白图

    new_img = img[tempr0:tempr1, tempc0:tempc1]
    #二值化
    retval,binary = cv2.threshold(new_img,0,255,cv2.THRESH_OTSU)

    #叠加两幅图像
    rstImg=ImageOverlay(imageTemp, binary)
    return rstImg

#字符图像的特征提取方法
#要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
def SimpleGridFeature(image):
    '''
    @description:提取字符图像的简单网格特征
    @image:灰度字符图像
    @return:长度为64字符图像的特征向量feature
    @author:
    '''
    new_img = HandWhiteEdges(image)  # 白边处理
    #图像大小归一化
    image = cv2.resize(new_img,(64,64))
    img_h = image.shape[0]
    img_w = image.shape[1]

    #二值化
    retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    #计算网格大小
    grid_size=16
    grid_h = binary.shape[0]/grid_size
    grid_w = binary.shape[1]/grid_size
    #定义特征向量
    feature = np.zeros(grid_size*grid_size)
    for j in range(grid_size):
        for i in range(grid_size):
            grid = binary[int(j*grid_h):int((j+1)*grid_h),int(i*grid_w):int((i+1)*grid_w)]
            feature[j*grid_size+i] = grid[grid==0].size
    return feature


def test_one_image(jpg_path):
    print("进入模型")
    with tf.Graph().as_default():  #
        output_graph_def = tf.GraphDef()  #
        pb_file_path = r"D:\\sxl\\VisualStudio\\CallTensorFlow2\\x64\Debug\\OCR.pb"  #模型路径
        with open(pb_file_path, "rb") as f:   #读取模型
            output_graph_def.ParseFromString(f.read())  # rb
            _ = tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            
            #我保存OCR模型时，将输入的名称命名为inputs，将输出命名为outputs,这个需要根据自己的代码更改！
            #也是成败的关键！
            input_x = sess.graph.get_tensor_by_name("inputs:0")  ####这个和之前保存模型时说的一样
            print(input_x)
            out_label = sess.graph.get_tensor_by_name("outputs:0")####这个和之前保存模型时说的一样
            print(out_label)
            print("开始读图")
            #这个npy文件，我是为了读取我需要的字符数组，可以不管它
            char_set = np.load(r"E:/sxl_Programs/Python/ANN/npy/ImgHanZiName653.npy")
            char_set = char_set.tolist()  #将读取的字符数组转化成列表
            # print(char_set)

            img = cv_imread(jpg_path, 0) #读图
            feature = SimpleGridFeature(img).reshape(-1, 256) #将图片转换成256*1的列表（提取特征用的）

            # plt.figure("fig1")
            # plt.imshow(img)
            img = img * (1.0 / 255)
            img_out_softmax = sess.run(out_label, feed_dict={input_x: feature}) #预测，返回的是x维的float数据，x是你保存模型时定义的输出维度

            # print ("img_out_softmax:",img_out_softmax)
            prediction_labels = np.argmax(img_out_softmax, axis=1) #返回该列表中最大值的下标
            print("prediction_labels:", prediction_labels)
            print("prediction_char:", char_set[prediction_labels[0]])#根据下标，输出该下标对应的字符
            # plt.show()

    print("结束！")


