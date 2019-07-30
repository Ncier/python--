#import cv2
#import  numpy as np
#img=cv2.imread(r'C:\Users\Administrator\Desktop\unet-v1-improvement-050-0.9867-Accu-0.9910Pred.jpg')
#
#
#ty=cv2.imshow('img', img)

#
#from PIL import Image
#import numpy as np
#import matplotlib.pyplot as plt
#img=np.array(Image.open(r'C:\Users\Administrator\Desktop\unet-v1-improvement-050-0.9867-Accu-0.9910Pred.jpg').convert('L'))
##img=img/255
#plt.figure("lena")
#arr=img.flatten()
#n, bins, patches = plt.hist(arr, bins=100, normed=2, facecolor='green', alpha=0.75)  
#plt.show()




from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img=np.array(Image.open(r'C:\Users\Administrator\Desktop\0617图像.png').convert('L'))

plt.figure("lena")
arr=img.flatten()
n, bins, patches = plt.hist(arr, bins=50, normed=1, facecolor='green', alpha=0.75)  
plt.show()