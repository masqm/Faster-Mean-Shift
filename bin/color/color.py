import cv2 
import numpy as np

import os
from libtiff import TIFF

from random import shuffle

img1 = cv2.imread('cpu.tif',-1)
img2 = cv2.imread('gpu.tif',-1)

height = img2.shape[0]
width = img2.shape[1]

cd = dict()

img3 = np.zeros((height,width,3), np.uint8)

for i in range(height):
    for j in range(width):
        if img2[i,j] != 0:
            if img1[i,j].any() != 0:
                if img2[i,j] not in cd:
                    cd.update({img2[i,j]:img1[i,j]})

for i in range(height):
    for j in range(width):
        if img2[i,j] != 0:
            img3[i,j] = cd.get(img2[i,j])

cv2.imwrite('out.tif',img3)
test = 0