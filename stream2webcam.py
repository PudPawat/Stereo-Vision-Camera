import cv2
import numpy as np
import json
import os
import glob
from pathlib import Path

###https://nerian.com/support/calculator/

### https://learnopencv.com/depth-perception-using-stereo-camera-python-c/#from-disparity-map-to-depth-map


def nothing(x):
    pass


class SetUpDisParity():
    def __init__(self,pathtofile = "calib/2/stereoMap.txt", setupstereoile = "sample.json"):
        self.stereo = cv2.StereoBM_create(numDisparities=16, blockSize=55)
        self.get_cvfile(pathtofile)

        try:
            self.setup_stereo_from_file(setupstereoile)
        except:
            pass

        self.set_setupwindow()

    def get_cvfile(self,pathtofile = "calib/2/stereoMap.txt"):
        self.stereo = cv2.StereoBM_create(numDisparities=16, blockSize=55)
        cv_file = cv2.FileStorage(pathtofile, cv2.FILE_STORAGE_READ)
        print(cv_file)
        self.Left_Stereo_Map_x = cv_file.getNode("stereoMapL_x").mat()
        self.Left_Stereo_Map_y = cv_file.getNode("stereoMapL_y").mat()
        self.Right_Stereo_Map_x = cv_file.getNode("stereoMapR_x").mat()
        self.Right_Stereo_Map_y = cv_file.getNode("stereoMapR_y").mat()
        cv_file.release()


    def set_setupwindow(self):
        cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('disp', 600, 600)
        try:

            cv2.createTrackbar('numDisparities', 'disp', self.parameters_from_file["numDisparities"], 100, nothing)
            cv2.createTrackbar('blockSize', 'disp', self.parameters_from_file["blockSize"], 50, nothing)
            cv2.createTrackbar('preFilterType', 'disp', self.parameters_from_file["preFilterType"], 1, nothing)
            cv2.createTrackbar('preFilterSize', 'disp', self.parameters_from_file["preFilterSize"], 25, nothing)
            cv2.createTrackbar('preFilterCap', 'disp', self.parameters_from_file["preFilterCap"], 62, nothing)
            cv2.createTrackbar('textureThreshold', 'disp', self.parameters_from_file["textureThreshold"], 100, nothing)
            cv2.createTrackbar('uniquenessRatio', 'disp', self.parameters_from_file["uniquenessRatio"], 100, nothing)
            cv2.createTrackbar('speckleRange', 'disp', self.parameters_from_file["speckleRange"], 200, nothing)
            cv2.createTrackbar('speckleWindowSize', 'disp', self.parameters_from_file["speckleWindowSize"], 200, nothing)
            cv2.createTrackbar('disp12MaxDiff', 'disp', self.parameters_from_file["disp12MaxDiff"], 200, nothing)
            cv2.createTrackbar('minDisparity', 'disp', self.parameters_from_file["minDisparity"], 200, nothing)

        except:
            cv2.createTrackbar('numDisparities', 'disp', 1, 100, nothing)
            cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
            cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
            cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
            cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
            cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
            cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
            cv2.createTrackbar('speckleRange', 'disp', 0, 200, nothing)
            cv2.createTrackbar('speckleWindowSize', 'disp', 3, 200, nothing)
            cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 200, nothing)
            cv2.createTrackbar('minDisparity', 'disp', 5, 200, nothing)

    def setup_stereo_from_file(self, setupfilepath = "sample.json"):

        with Path(setupfilepath).open("r") as f:
            self.parameters_from_file = json.load(f)

        self.stereo.setNumDisparities(self.parameters_from_file["numDisparities"])
        self.stereo.setBlockSize(self.parameters_from_file["blockSize"])
        self.stereo.setPreFilterType(self.parameters_from_file["preFilterType"])
        self.stereo.setPreFilterSize(self.parameters_from_file["preFilterSize"])
        self.stereo.setPreFilterCap(self.parameters_from_file["preFilterCap"])
        self.stereo.setTextureThreshold(self.parameters_from_file["textureThreshold"])
        self.stereo.setUniquenessRatio(self.parameters_from_file["uniquenessRatio"])
        self.stereo.setSpeckleRange(self.parameters_from_file["speckleRange"])
        self.stereo.setSpeckleWindowSize(self.parameters_from_file["speckleWindowSize"])
        self.stereo.setDisp12MaxDiff(self.parameters_from_file["disp12MaxDiff"])
        self.stereo.setMinDisparity(self.parameters_from_file["minDisparity"])


    def get_disparity(self,imgL_calib, imgR_calib):
        # Calculating disparity using the StereoBM algorithm
        disparity = self.stereo.compute(imgL_calib, imgR_calib)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_32F and scale it down 16 times.

        # Converting to float32
        disparity = disparity.astype(np.float32)
        return  disparity


    def get_setup_stereo(self, imgL, imgR):

        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        Left_nice = cv2.remap(imgL_gray,
                              self.Left_Stereo_Map_x,
                              self.Left_Stereo_Map_y,
                              cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT, 0)

        Right_nice = cv2.remap(imgR_gray,
                               self.Right_Stereo_Map_x,
                               self.Right_Stereo_Map_y,
                               cv2.INTER_LANCZOS4,
                               cv2.BORDER_CONSTANT,
                               0)


        # Updating the self.parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

        # Setting the updated self.parameters before computing disparity map
        self.parameters["numDisparities"] = int(numDisparities /16)
        self.parameters["blockSize"] = int((blockSize -5 ) /2)
        self.parameters["preFilterType"] = preFilterType
        self.parameters["preFilterSize"] = int((preFilterSize -5 )/2)
        self.parameters["preFilterCap"] = preFilterCap
        self.parameters["textureThreshold"] = textureThreshold
        self.parameters["uniquenessRatio"] = uniquenessRatio
        self.parameters["speckleRange"] = speckleRange
        self.parameters["speckleWindowSize"] = int(speckleWindowSize /2)
        self.parameters["disp12MaxDiff"] = disp12MaxDiff
        self.parameters["minDisparity"] = minDisparity

        if numDisparities <= 1:
            numDisparities = 2
        self.stereo.setNumDisparities(numDisparities)
        self.stereo.setBlockSize(blockSize)
        self.stereo.setPreFilterType(preFilterType)
        self.stereo.setPreFilterSize(preFilterSize)
        self.stereo.setPreFilterCap(preFilterCap)
        self.stereo.setTextureThreshold(textureThreshold)
        self.stereo.setUniquenessRatio(uniquenessRatio)
        self.stereo.setSpeckleRange(speckleRange)
        self.stereo.setSpeckleWindowSize(speckleWindowSize)
        self.stereo.setDisp12MaxDiff(disp12MaxDiff)
        self.stereo.setMinDisparity(minDisparity)

        disparity = self.get_disparity(Left_nice,Right_nice)

        # Scaling down the disparity values and normalizing them
        # disparity = ((disparity - min_dis) / (max_dis - min_dis) * 255).astype(np.uint8)
        # disparity = ((disparity / 16.0 - minDisparity) / numDisparities)
        # disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

        # Filter Disparity Estimate
        local_max = disparity.max()
        local_min = disparity.min()
        disparity_grayscale = (disparity - local_min) * (65535.0 / (local_max - local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
        disparity_show = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

        return disparity_show, disparity_fixtype, disparity
        
    def setup_disparity_streaming_webcam(self,readL,readR):
        i = 0
        self.parameters = {}

        while True:
            retL, imgL = readL.read()
            retR, imgR = readR.read()

            cv2.imshow('img left1', imgL)
            cv2.imshow('img right1', imgR)

            # Proceed only if the frames have been captured
            if retL and retR:
                
                disparity,_,disparity_original = self.get_setup_stereo(imgL,imgR)
                cv2.imshow("disp", disparity)

                norm_image = cv2.normalize(disparity_original, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                norm_image = cv2.convertScaleAbs(norm_image, alpha=255.0)
                disparity_show = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)
                cv2.imshow("disp1", disparity_show)

                depth = self.convert2depth(norm_image, 0.015)
                local_min = depth.min()
                local_max = depth.max()
                print("depth_min max", local_min, local_max)
                # depth_grayscale = (depth - local_min)  / (local_max - local_min)
                depth = cv2.normalize(depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                           dtype=cv2.CV_32F)
                depth_fixtype = cv2.convertScaleAbs(depth, alpha=255)
                disparity_show = cv2.applyColorMap(depth_fixtype, cv2.COLORMAP_JET)

                cv2.imshow("depth",disparity_show)


                key = cv2.waitKey(1)

                if key == ord("c"):
                    cv2.imwrite("calib/2/image_L_00{}.jpg".format(i), imgL)
                    cv2.imwrite("calib/2/image_R_00{}.jpg".format(i), imgR)

                    i +=1

                elif key == ord("s"):
                    print(self.parameters)
                    with open("sample.json", "w") as outfile:
                        json.dump(self.parameters, outfile)


    def convert23D(self,disparity, Q):
        Q = np.asarray(Q)
        depth = cv2.reprojectImageTo3D(disparity, Q)

        return depth

    def convert2depth(self,disparity, baseline_dis, focal = 18):

        convergence_dis = 1
        inv_convergence_dis = 1/convergence_dis
        sensor_size = 0.008819 ## in meter
        width_resolution = 640
        pixel_width = sensor_size/width_resolution
        focal_length = 0.018 ### 18 mm
        inter_axial_dis = 0.015 ## 15 cm distance between camera (measure form center of len to len)
        factor_of_combining_phy_param = focal_length* inter_axial_dis / pixel_width
        adjusted_disparity = 2* ( -1*disparity/factor_of_combining_phy_param)
        real_distance = 1/(inv_convergence_dis - adjusted_disparity)




        # depth = baseline_dis*focal*np.linalg.inv(disparity)
        # depth = baseline_dis*focal/disparity
        return real_distance

if __name__ == '__main__':
    readR = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    readL = cv2.VideoCapture(2, cv2.CAP_DSHOW)


    Setup = SetUpDisParity()
    Setup.setup_disparity_streaming_webcam(readL, readR)