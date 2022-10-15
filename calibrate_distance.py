import cv2
import numpy as np
import os
import glob

from stream2webcam import SetUpDisParity




if __name__ == '__main__':

    readR = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    readL = cv2.VideoCapture(2, cv2.CAP_DSHOW)


    Setup = SetUpDisParity()
    # Setup.setup_disparity_streaming_webcam(readL, readR)

    # image = cv2.imread("image.jpg")


    cv2.namedWindow('disp1')
    cv2.setMouseCallback('disp1', Setup.mouseRGB)



    i = 0
    Setup.parameters = {}

    while True:
        retL, imgL = readL.read()
        retR, imgR = readR.read()

        # self.imgL = imgL

        cv2.imshow('img left1', imgL)
        cv2.imshow('img right1', imgR)

        # Proceed only if the frames have been captured
        if retL and retR:
            disparity, _, disparity_original = Setup.get_setup_stereo(imgL, imgR)

            cv2.imshow("disp", disparity)
            cv2.imshow("origianl disparity", disparity_original)
            Setup.disparity_original = disparity_original

            norm_image = cv2.normalize(disparity_original, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)

            norm_image = cv2.convertScaleAbs(norm_image, alpha=255.0)
            disparity_show = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)
            cv2.imshow("disp1", disparity_show)

            # depth = Setup.convert2depth(disparity_original, 0.0105)
            depth = Setup.convert2depth_by_coeff(disparity_original)

            cv2.waitKey(1)
