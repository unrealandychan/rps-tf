import cv2
from rpscv import utils
from rpscv import imgproc as imp

from keras.models import load_model
import numpy as np
import time

filename = 'rps.h5'
model = load_model(filename)

cam = utils.cameraSetup()
cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)

result = {
    0:"Rock",
    1:"Paper",
    2:"Scissors"
}


while True:
    img = cam.getOpenCVImage()
    img = imp.crop(img)
    cv2.imshow('Camera', img)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resizeimg = np.array(imgRGB).reshape((-1, 200, 200, 3))

    predGesture = model.predict_classes(resizeimg)[0]

    print(result[predGesture])