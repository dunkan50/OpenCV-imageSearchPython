import numpy as np
import cv2

class colorsDescriptors:
    def __init__(self, bins):
        self.bins = bins

    def explain(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        features = []
        (l, b) = image.shape[:2]
        (cX, cY) = (int(b*0.5), int(l*0.5))
        parts = [(0, cX, 0, cY), (cX, b, 0, cY), (cY, b, cY, l), (0, cX, cY,l)]

        (axisX, axisY) = (int(b*0.75)/2, int(l*0.75)/2)
        ellipseMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipseMask, (cX, cY), (axisX, axisY), 0, 0, 360, 255, -1)
        for (Xstart, Xend, Ystart, Yend) in parts:
            corners = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(corners, (Xstart, Ystart), (Xend, Yend), 255, -1)
            corners = cv2.subtract(corners, ellipseMask)
            histo = self.histogram(image, corners)
            features.extend(histo)

        histo = self.histogram(image, ellipseMask)
        features.extend(histo)
        return features

    def histogram(self, image, mask):
        histo = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256,0, 256])
        cv2.normalize(histo, histo)
        histo= histo.flatten()
        return histo

