import cv2
import numpy as np
import random as rng


im=cv2.imread("training_chars.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray,200 , 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_draw=cv2.drawContours(im, contours, -1, (0,0,0), 1)
count=0
crop_list=[]
for c in contours:
    count+=1
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a rectangle to visualize the bounding rect

    bounded_img=cv2.rectangle(imgray, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("img,",bounded_img)
    crop_list.append(cv2.resize(im[y:y+h, x:x+w],(30,30),interpolation=cv2.INTER_AREA))
d=0

for i in crop_list:
    filename = "images/file_%d.jpg"%d
    cv2.imwrite(filename, i)
    d+=1
    

        