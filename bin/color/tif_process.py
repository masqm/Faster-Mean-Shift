import cv2 
import numpy as np

import os
from libtiff import TIFF

from random import shuffle

path = "E:\\code\\gpu\\in" #文件夹目录
out = "E:\\code\\gpu\\out" #文件夹目录


colors = [(255,0,0), (0,255,0),(0,0,255),(0,255,255),(255,255,0),(255,0,255), (255,51,0),(51,255,0),(204,51,51),(255,255,153),(255,153,0),(204,0,0),(255,153,102),(255,102,0),(255,102,102),(255,153,153),
          (0,51,255),(0, 255,51),(51,204,51),(153,255,255),(0,255,153),(0,204,0),(102,255,153),(0,255,102),(102,255,102),(153,255,153),
          (51,0,255),(255,0,51),(51,51,204),(255,153,255),(153,0,255),(0,0,204),(153,102,255),(102,0,255),(102,102,255),(153,153,255),
          (255,255,255)]

def manipu_cv(img, label):
    
    height = img.shape[0]
    width = img.shape[1]

    img2 = np.zeros((height,width,3), np.uint8)
    for i in range(height):
        for j in range(width):
            #if img[i,j]%2 == 0:
            #    img[i,j] +=1

            if img[i,j] != 0:
                img2[i,j]=colors[int(img[i,j]%7)]#37

            #if img[i,j] == 18 or img[i,j] == 19:
            #    img[i,j]=0
            #else:
            #    img[i,j]=255
    return img2


def manipu_cv2(img):
    
    height = img.shape[0]
    width = img.shape[1]

    img2 = np.zeros((height,width,3), np.uint8)
    for i in range(height):
        for j in range(width):

            if img[i,j] != 0:
                img[i,j]=255

    return img


if __name__ == "__main__":

    #files= os.listdir(path)
    #files = os.walk(path)

    shuffle(colors)

    files=[]
    fpathes =[]
    for fpathe,dirs,fs in os.walk(path):
        fpathes.append(fpathe)
        for f in fs:
            files.append(os.path.join(fpathe,f))

    for folder in fpathes[1:]:
        os.mkdir(folder.replace("\\in","\\out"))



    for file in files: #遍历文
        position = file
        out_position = file.replace("\\in","\\out")
        print (file)

        img1 = cv2.imread(position,-1)

        img = manipu_cv(img1,0)
        retval, dst = cv2.threshold(img1, 0, 65535, 0)

        #cv2.imshow('binary', dst)
        #cv2.waitKey(0)

        cv2.imwrite(out_position,img)
        test = 0


    
    #label=[]
    #position = []
    #out_position = []

    #for file in files: #遍历文件夹
    #    position.append(file)
    #    out_position.append(file.replace("\\in","\\out"))
    #    print (file)

    #img1 = cv2.imread(position[0],-1)
    #img2 = cv2.imread(position[1],-1)

    #p_set = np.unique(img1)
    #label = np.unique(np.append(label,p_set))

    #p_set = np.unique(img2)
    #label = np.unique(np.append(label,p_set))

    #img = manipu_cv(img1, label)
    #cv2.imwrite(out_position[0].replace(".png", ".tif"),img)

    #img = manipu_cv(img2, label)
    #cv2.imwrite(out_position[1].replace(".png", ".tif"),img)

    #test=0



        #img = cv2.imread(position,-1)

        #cv2.namedWindow("Image")
        #cv2.imshow("Image",img)
        #cv2.waitKey(0)
        ##释放窗口
        #cv2.destroyAllWindows() 


        #p_set = np.unique(img)

        #label = np.unique(np.append(label,p_set))
        

        #img2 = manipu_cv(img)

        #test =1
        #cv2.imwrite(out_position.replace(".png", ".tif"),img2)


        #test =1


