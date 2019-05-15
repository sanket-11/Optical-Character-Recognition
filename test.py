# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:27:41 2019

@author: Sanket Patole
"""

import cv2
import numpy as np
import random as rng
import os, sys
import tensorflow as tf

#read test image
im=cv2.imread("test1.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray,200 , 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_draw=cv2.drawContours(im, contours, -1, (0,0,0), 1)
#cv2.imshow("img",contour_draw)
count=0
crop_list=[]
for c in contours:
    count+=1
    #ignore first contour which is the frame of image
    if count>1:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a rectangle to visualize the bounding rect
    
        bounded_img=cv2.rectangle(imgray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        crop_image=cv2.resize(im[y:y+h, x:x+w],(30,30),interpolation=cv2.INTER_AREA)
    
        _,convert=cv2.imencode('.png', crop_image)
        byte_image=convert.tobytes()
    
    
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    
    
    # Read image_data in bytes
        image_data = byte_image
        
        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line 
                           in tf.gfile.GFile("retrained_labels.txt")]
        
        # Unpersists graph from file
        with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        scores_dict=dict()
        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
            predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                # print('%s (score = %.5f)' % (human_string, score))
                scores_dict[human_string]=score
        max_score_label=max(scores_dict, key=scores_dict.get)
        max_value = max(scores_dict.values())
        # print(max_score_label,max_value)
        if max_value>0.3:
        	abc=cv2.putText(im, max_score_label, (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA) 

cv2.imwrite("output.png",abc)

               