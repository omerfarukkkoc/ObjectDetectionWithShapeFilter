# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:42:21 2017

@author: omerf
"""

import cv2
import numpy as np
import sys
import time
from shapedetector import ShapeDetector

sd = ShapeDetector()

def nothing(x):
    pass

fps = 0

cap = cv2.VideoCapture(0)

hue_min = 157
saturation_min = 149
value_min = 53
resize_width = 1024
resize_height = 768
location_sensitivity = 5
# # # # # # variables # # # # # #

hue_max = 179
saturation_max = 255
value_max = 255
cv2.namedWindow('Renk Ayar Penceresi')
cv2.createTrackbar('H-Min', 'Renk Ayar Penceresi', 0, 179, nothing)
cv2.createTrackbar('S-Min', 'Renk Ayar Penceresi', 0, 255, nothing)
cv2.createTrackbar('V-Min', 'Renk Ayar Penceresi', 0, 255, nothing)

cv2.setTrackbarPos('H-Min', 'Renk Ayar Penceresi', hue_min)
cv2.setTrackbarPos('S-Min', 'Renk Ayar Penceresi', saturation_min)
cv2.setTrackbarPos('V-Min', 'Renk Ayar Penceresi', value_min)

color_option_frame = np.zeros((1, 350, 3), np.uint8)

erode_matrix = np.ones((3, 3), np.uint8)
dilate_matrix = np.ones((3, 3), np.uint8)

if cap.isOpened():
    print('Kamera Açıldı')
else:
    print('HATA!! \nKamera Açılamadı!!')
    exit(1)

frame_count = 0
while 1:

    try:
        start = time.time()
        cv2.imshow('Renk Ayar Penceresi', color_option_frame)
        ret, frame = cap.read()

        if not ret:
            print('HATA!! Frame Alınamıyor \nYeniden Başlatın')
            cv2.destroyAllWindows()
            cap.release()
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hue_min = cv2.getTrackbarPos('H-Min', 'Renk Ayar Penceresi')
        saturation_min = cv2.getTrackbarPos('S-Min', 'Renk Ayar Penceresi')
        value_min = cv2.getTrackbarPos('V-Min', 'Renk Ayar Penceresi')

        lower_limit = np.array([hue_min, saturation_min, value_min], dtype=np.uint8)
        upper_limit = np.array([hue_max, saturation_max, value_max], dtype=np.uint8)

        threshold = cv2.inRange(hsv_frame, lower_limit, upper_limit)
        threshold = cv2.erode(threshold, erode_matrix)
        threshold = cv2.dilate(threshold, dilate_matrix)

        (im2, contours, hierarchy) = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            print(cv2.contourArea(c))
            if cv2.contourArea(c) < 500:
                continue
            shape = sd.detect(c)
            # print(shape)

            approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
            if shape == "circle":
                cv2.putText(frame, shape, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.drawContours(frame, [c], 0, (0, 255, 0), 5)
            # print(len(approx))
            # (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, "Fps: "+str(fps), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        threshold = cv2.resize(threshold, (640, 480), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('thresh', threshold)
        cv2.imshow('frame', frame)
        frame_count += 1
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            print("Çıkış Yapıldı")
            break

        fps = np.float16((1 / (time.time() - start)))

    except:
        print("Beklenmedik Hata!!! ", sys.exc_info()[0])
        raise

cv2.destroyAllWindows()
cap.release()