import numpy as np
import cv2
import glob
# Check for left and right camera IDs
# These values can change depending on the system
# CamL_id = 2  # Camera ID for left camera
# CamR_id = 0  # Camera ID for right camera
#
# CamL = cv2.VideoCapture(CamL_id)
# CamR = cv2.VideoCapture(CamR_id)




def nothing(x):
    pass

def setup_disparity(imagesLeftDir, imagesRightDir, stereo):
    '''

    :param imagesLeftDir:
    :param imagesRightDir:
    :param stereo: object
    :return:
    '''
    # while True:
    for imgLeft, imgRight in zip(imagesLeftDir, imagesRightDir):
        while True:
            print("here")
            # Capturing and storing left and right camera images
            # retL, imgL = CamL.read()
            # retR, imgR = CamR.read()
            imgL = cv2.imread(imgLeft)
            imgR = cv2.imread(imgRight)
            # print(imgR.shape)
            cv2.imshow('img left1', imgL)
            # cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            cv2.imshow('img right1', imgR)
            cv2.waitKey(1)

            # imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            # imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            print(imgR.shape)
            print(imgL.shape)


            # Proceed only if the frames have been captured
            # if retL and retR:
            imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

            Left_nice = cv2.remap(imgL_gray,
                                  Left_Stereo_Map_x,
                                  Left_Stereo_Map_y,
                                  cv2.INTER_LANCZOS4,
                                  cv2.BORDER_CONSTANT,0)

            print(Left_nice.shape)
            # Applying stereo image rectification on the right image
            Right_nice = cv2.remap(imgR_gray,
                                   Right_Stereo_Map_x,
                                   Right_Stereo_Map_y,
                                   cv2.INTER_LANCZOS4,
                                   cv2.BORDER_CONSTANT,
                                   0)

            cv2.imshow('img left1', Left_nice)
            # cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            cv2.imshow('img right1', Right_nice)
            cv2.waitKey(1)

            # Updating the parameters based on the trackbar positions
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

            # Setting the updated parameters before computing disparity map
            stereo.setNumDisparities(numDisparities)
            stereo.setBlockSize(blockSize)
            stereo.setPreFilterType(preFilterType)
            stereo.setPreFilterSize(preFilterSize)
            stereo.setPreFilterCap(preFilterCap)
            stereo.setTextureThreshold(textureThreshold)
            stereo.setUniquenessRatio(uniquenessRatio)
            stereo.setSpeckleRange(speckleRange)
            stereo.setSpeckleWindowSize(speckleWindowSize)
            stereo.setDisp12MaxDiff(disp12MaxDiff)
            stereo.setMinDisparity(minDisparity)

            # Calculating disparity using the StereoBM algorithm
            disparity = stereo.compute(Left_nice, Right_nice)
            # NOTE: Code returns a 16bit signed single channel image,
            # CV_16S containing a disparity map scaled by 16. Hence it
            # is essential to convert it to CV_32F and scale it down 16 times.

            # Converting to float32
            disparity = disparity.astype(np.float32)

            print(np.max(disparity))
            print(np.min(disparity))
            min_dis = np.min(disparity)
            max_dis = np.max(disparity)

            # Scaling down the disparity values and normalizing them
            disparity = ((disparity-min_dis)/(max_dis-min_dis) *255).astype(np.uint8)
            # disparity = ((disparity / 16.0 - minDisparity) / numDisparities)


            # disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
            # Displaying the disparity map
            cv2.imshow("disp", disparity)

            # Close window using esc key
            if cv2.waitKey(0) == 27:
                break


if __name__ == '__main__':
    # Reading the mapping values for stereo image rectification
    cv_file = cv2.FileStorage("calib/2/stereoMap.txt", cv2.FILE_STORAGE_READ)
    print(cv_file)
    Left_Stereo_Map_x = cv_file.getNode("stereoMapL_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("stereoMapL_y").mat()
    Right_Stereo_Map_x = cv_file.getNode("stereoMapR_x").mat()
    Right_Stereo_Map_y = cv_file.getNode("stereoMapR_y").mat()
    cv_file.release()

    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 600, 600)

    cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
    cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
    cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
    cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
    cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
    cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
    cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
    cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
    cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
    cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)

    # Creating an object of StereoBM algorithm
    stereo = cv2.StereoBM_create()

    imagesLeft = sorted(glob.glob('F:\github\stereo_vision_scratch\calib\\2/image_L*'))
    imagesRight = sorted(glob.glob('F:\github\stereo_vision_scratch\calib/2/image_R*'))
