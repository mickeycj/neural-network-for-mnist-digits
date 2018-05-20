import cv2
import numpy as np

img_path = "/home/wisarut/Desktop/picture1.png"

im = cv2.imread(img_path)

im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray,(5,5),0)

ret, im_th = cv2.threshold(im_gray,90,255,cv2.THRESH_BINARY_INV)

_,ctrs, hier = cv2.findContours(im_th.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(ctr) for ctr in ctrs]
i = 10
for rect in rects:
  # #Draw the rectangles
  # cv2.rectangle(im, (rect[0],rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

  # #Make the rectangles
  # leng = int(rect[3] * 1.6)
  # pt1 = int(rect[1] + rect[3] // 2- leng // 2)
  # pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
  # roi = im_th[pt1:pt1 + leng, pt2:pt2+leng]

  # roi = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
  # roi = cv2.dilate(roi, (3,3))

  i = i -1
  x, y, w, h = cv2.boundingRect(ctr)
  # Getting ROI
  leng = int(rect[3] * 1.6)
  pt1 = int(rect[1] + rect[3] // 2- leng // 2)
  pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
  roi = im_th[pt1:pt1 + leng, pt2:pt2+leng]
  roi = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
  roi = cv2.dilate(roi, (3,3))

  # show ROI
  cv2.imshow( str(pt1) +',' + str(pt2) +' : '+str(i),roi)
  cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 1)
  cv2.waitKey(0)
  
  # save only the ROI's which contain a valid information
  if h > 10 and w > 10:
    cv2.imwrite('roi\\{}.png'.format(i), roi) 


cv2.imshow("ReSulting image woth rectangular rois", im)
cv2.waitKey()