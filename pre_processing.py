import cv2
import numpy as np

img_path = "./id/test.png"

#read image
img = cv2.imread(img_path)

# read the image in grayscale 
img_gray = cv2.imread(img_path, 0)
cv2.imshow("load image grayscale", img_gray)
cv2.waitKey()

# Apply Gaussain filtering
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

# binary in gray
ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV) 
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

# dilation
kernel = np.ones((3, 3), np.uint8)  # values set for this image only - need to change for different images
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow("dilated", img_dilation)
cv2.waitKey(0)

# find contours
_, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours from x + y
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1])

for i, rect in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(rect)

    # Getting ROI and resize image
    leng = int(h * 1.6)
    pt1 = int(y + h // 2- leng // 2)
    pt2 = int(x + w // 2 - leng // 2)
    roi = img_dilation[pt1:pt1 + leng, pt2:pt2+leng]
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # show ROI
    cv2.imshow(str(pt1) + "," + str(pt2) + " : " + str(i), roi)
    cv2.waitKey(0)

    #write image 
    cv2.imwrite("roi\\{}.png".format(i), roi) 
