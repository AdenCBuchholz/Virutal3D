# This will be an object oriented version
# of the virtual3d game
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Facefinder:
#Use haarcascade filter to detect largest face from a frame.
    def __init__ (self):
        print ('Face Finder Initalize')
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def find_face(self, frame):
        """Returns face center (x,y). Draws rectangle on frame"""
        # convert to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, minNeighbors = 9)
        #Draw Rectangle
        if faces is None:
            return(None)
        bx = by = bw = bh = 0
             
        for (x, y, w, h) in faces:
            if w > bw:
              bx, by, bw, bh = x, y, w, h  
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 255), 5)
            return(bx + bw/2), (by + bh//2)
#---------------------------------------------------------------
# Main
ff = Facefinder()
print("Virtual3d Complete")