import cv2
import numpy as np

def eaualHist(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    
    #cv2.namedWindow('input_image', cv.WINDOW_NORMAL)
    #cv2.imshow('input_image', gray)

    gray = np.array(image, dtype=np.uint8)
    dst = cv2.equalizeHist(gray)              
    #cv2.namedWindow("eaualHist_demo", cv2.WINDOW_NORMAL)
    #cv2.imshow("eaualHist_demo", dst)
    return dst


if __name__ == '__main__':

    output_img = "E:/code/data/myfile/out/mask000.tif"

    src=cv2.imread(output_img, -1)

    dst = eaualHist(src)

    cv2.imshow("eaualHist", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




