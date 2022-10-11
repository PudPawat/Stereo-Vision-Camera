import cv2
import HSV_filter as hsv

cap = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_SETTINGS,1)
cap2.set(cv2.CAP_PROP_SETTINGS,1)

num = 0

while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()


    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('C:/Users\Air lab/.spyder-py3/stereo calibration/point cloud/stereoLeft/imageL0' + str(num) + '.png', img)
        cv2.imwrite('C:/Users\Air lab/.spyder-py3/stereo calibration/point cloud/stereoright/imageR0' + str(num) + '.png', img2)
        print("images saved!")
        num += 1

    cv2.imshow('Img L',img)
    cv2.imshow('Img R',img2)

# Release and destroy all windows before termination
cap.release()
cap2.release()

cv2.destroyAllWindows()