import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob

imagesLeft = sorted(glob.glob('F:\github\stereo_vision_scratch\calib\\image_L*'))
imagesRight = sorted(glob.glob('F:\github\stereo_vision_scratch\calib/image_R*'))



imgL = cv.imread('tsukuba_l.png',0)
imgR = cv.imread('tsukuba_r.png',0)
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()


