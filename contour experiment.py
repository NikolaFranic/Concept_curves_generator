import cv2 as cv
import numpy as np

img = cv.imread("rs6_side_view.jpg")
cv.imshow("Side_view", img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow("grey", gray)

threshold, thresh = cv.threshold(gray,55,255,cv.THRESH_BINARY)
#cv.imshow("thres", thresh)

lower_red = np.array([0,0,100])
upper_red = np.array([80,80,255])
mask = cv.inRange(img,lower_red,upper_red)

cv.imshow("mask", mask)

cv.waitKey(0)