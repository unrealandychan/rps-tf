from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import tensorflow as tf

img = cv2.imread("/Users/anserbridge/PycharmProjects/rps-cv/rps-cv/train_img/paper/DSC01996.jpg")

print(np.shape(img))

img_resize = np.array(img).reshape((-1,4000,6000,3))
print(img_resize.shape)