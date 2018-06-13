#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:43:54 2018

@author: vladimir
"""

import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.15,
#    'gpu': 1.0
}

tfnet = TFNet(option)

capture = cv2.VideoCapture('HONEY_BEES_96fps_IN_4K_(ULTRA_HD).mp4')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    if ret:
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label'] + ': ' + str(int(result['confidence']*100)) + '%'
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break