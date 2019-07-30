
import scipy.misc as misc
import matplotlib
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def transfer_image(path,img):
    image=cv2.imread(path+img)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    label=np.zeros((image.shape[0],image.shape[1]))
   
    ## 水田1
    index_11=image[:,:,0]==0
    index_12=image[:,:,1]==200
    index_13=image[:,:,2]==0
    t=index_11*index_12
    resul1=t*index_13
    label[resul1]=1
    ## 水浇地2
    index_21=image[:,:,0]==150
    index_22=image[:,:,1]==250
    index_23=image[:,:,2]==0
    t=index_21*index_22
    resul2=t*index_23
    label[resul2]=2
    ## 旱耕地3
    index_31=image[:,:,0]==150
    index_32=image[:,:,1]==200
    index_33=image[:,:,2]==150
    t=index_31*index_32
    resul3=t*index_33
    label[resul3]=3
    ##园地4
    index_41=image[:,:,0]==200
    index_42=image[:,:,1]==0
    index_43=image[:,:,2]==200
    t=index_41*index_42
    resul4=t*index_43
    label[resul4]=4
    ##乔木林地5
    index_51=image[:,:,0]==150
    index_52=image[:,:,1]==0
    index_53=image[:,:,2]==250
    t=index_51*index_52
    resul5=t*index_53
    label[resul5]=5
    ##灌木林地6
    index_61=image[:,:,0]==150
    index_62=image[:,:,1]==150
    index_63=image[:,:,2]==250
    t=index_61*index_62
    resul6=t*index_63
    label[resul6]=6
   
   
    ##村镇住宅11
    index_111=image[:,:,0]==200
    index_112=image[:,:,1]==150
    index_113=image[:,:,2]==150
    t=index_111*index_112
    resul11=t*index_113
    label[resul11]=11
   
    ##交通运输12
    index_121=image[:,:,0]==250
    index_122=image[:,:,1]==150
    index_123=image[:,:,2]==150
    t=index_121*index_122
    resul12=t*index_123
    label[resul12]=12
    ##河流13
    index_131=image[:,:,0]==0
    index_132=image[:,:,1]==0
    index_133=image[:,:,2]==200
    t=index_131*index_132
    resul13=t*index_133
    label[resul13]=13
    ##湖泊14
    index_141=image[:,:,0]==0
    index_142=image[:,:,1]==150
    index_143=image[:,:,2]==200
    t=index_141*index_142
    resul14=t*index_143
    label[resul14]=14
    ##坑塘15
    index_151=image[:,:,0]==0
    index_152=image[:,:,1]==200
    index_153=image[:,:,2]==250
    t=index_151*index_152
    resul15=t*index_153
    label[resul15]=15
     
     ##天然草地7
    index_71=image[:,:,0]==250
    index_72=image[:,:,1]==200
    index_73=image[:,:,2]==0
    t=index_71*index_72
    resul7=t*index_73
    label[resul7]=7
    ##人工草地8
    index_81=image[:,:,0]==200
    index_82=image[:,:,1]==200
    index_83=image[:,:,2]==0
    t=index_81*index_82
    resul8=t*index_83
    label[resul8]=8
    ##工业用地9
    index_91=image[:,:,0]==200
    index_92=image[:,:,1]==0
    index_93=image[:,:,2]==0
    t=index_91*index_92
    resul9=t*index_93
    label[resul9]=9

     ##其它0
    index_01=image[:,:,0]==0
    index_02=image[:,:,1]==0
    index_03=image[:,:,2]==0
    t=index_01*index_02
    resul0=t*index_03
    label[resul0]=0
     ##城市住宅10
    index_101=image[:,:,0]==250
    index_102=image[:,:,1]==0
    index_103=image[:,:,2]==150
    t=index_101*index_102
    resul10=t*index_103
    label[resul10]=10
    cv2.imwrite(path+img.split('.')[0]+'-GT'+'.png',label)