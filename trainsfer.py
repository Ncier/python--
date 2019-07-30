import numpy as np
import  cv2
import os
from tqdm import tqdm

def transfer_image(path,img):
    image=cv2.imread(path+img)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    label=np.zeros((image.shape[0],image.shape[1]))
    ##其它0
    index_1=image[:,:,0]==0
    index_2=image[:,:,1]==0
    index_3=image[:,:,2]==0
    t=index_1==index_2
    resul0=t==index_3
    label[resul0]=0
    ## 水田1
    index_1=image[:,:,0]==0
    index_2=image[:,:,1]==200
    index_3=image[:,:,2]==0
    t=index_1==index_2
    resul1=t==index_3
    label[resul1]=1
    ## 水浇地2
    index_1=image[:,:,0]==150
    index_2=image[:,:,1]==250
    index_3=image[:,:,2]==0
    t=index_1==index_2
    resul2=t==index_3
    label[resul2]=2
    ## 旱耕地3
    index_1=image[:,:,0]==150
    index_2=image[:,:,1]==200
    index_3=image[:,:,2]==150
    t=index_1==index_2
    resul3=t==index_3
    label[resul3]=3
    ##园地4
    index_1=image[:,:,0]==200
    index_2=image[:,:,1]==0
    index_3=image[:,:,2]==200
    t=index_1==index_2
    resul4=t==index_3
    label[resul4]=4
    ##乔木林地5
    index_1=image[:,:,0]==150
    index_2=image[:,:,1]==0
    index_3=image[:,:,2]==250
    t=index_1==index_2
    resul5=t==index_3
    label[resul5]=5
    ##灌木林地6
    index_1=image[:,:,0]==150
    index_2=image[:,:,1]==150
    index_3=image[:,:,2]==250
    t=index_1==index_2
    resul6=t==index_3
    label[resul6]=6
    ##天然草地7
    index_1=image[:,:,0]==250
    index_2=image[:,:,1]==200
    index_3=image[:,:,2]==0
    t=index_1==index_2
    resul7=t==index_3
    label[resul7]=7
    ##人工草地8
    index_1=image[:,:,0]==200
    index_2=image[:,:,1]==200
    index_3=image[:,:,2]==0
    t=index_1==index_2
    resul8=t==index_3
    label[resul8]=8
    ##工业用地9
    index_1=image[:,:,0]==200
    index_2=image[:,:,1]==0
    index_3=image[:,:,2]==0
    t=index_1==index_2
    resul9=t==index_3
    label[resul9]=9

    ##城市住宅10
    index_1=image[:,:,0]==250
    index_2=image[:,:,1]==0
    index_3=image[:,:,2]==150
    t=index_1==index_2
    resul10=t==index_3
    label[resul10]=10
    ##村镇住宅11
    index_1=image[:,:,0]==200
    index_2=image[:,:,1]==150
    index_3=image[:,:,2]==150
    t=index_1==index_2
    resul11=t==index_3
    label[resul11]=11
    ##交通运输12
    index_1=image[:,:,0]==250
    index_2=image[:,:,1]==150
    index_3=image[:,:,2]==150
    t=index_1==index_2
    resul12=t==index_3
    label[resul12]=12
    ##河流13
    index_1=image[:,:,0]==0
    index_2=image[:,:,1]==0
    index_3=image[:,:,2]==200
    t=index_1==index_2
    resul13=t==index_3
    label[resul13]=13
    ##湖泊14
    index_1=image[:,:,0]==0
    index_2=image[:,:,1]==150
    index_3=image[:,:,2]==200
    t=index_1==index_2
    resul14=t==index_3
    label[resul14]=14
    ##坑塘15
    index_1=image[:,:,0]==0
    index_2=image[:,:,1]==200
    index_3=image[:,:,2]==250
    t=index_1==index_2
    resul15=t==index_3
    label[resul15]=15
    cv2.imwrite(path+img.split('.')[0]+'_transfer'+'.png',label)

for i in tqdm(os.listdir('./train/label_process/')):
    transfer_image('./train/label_process/',i)
for i in tqdm(os.listdir('./val/val_process')):
    transfer_image('./val/val_process/',i)