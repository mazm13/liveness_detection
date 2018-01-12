import cv2
import numpy as np
from lib import face
from lib.get_LBP_from_Image import LBP
from matplotlib import pyplot as plt
from lib.utils import read_path

import os
import numpy as np

posi_test = 'client_test_raw.txt'
nega_test = 'imposter_test_raw.txt'
 
paths = read_path(nega_test)
for path in paths[:10]:
    image = cv2.imread(path)
    gray = face.detect(image)

    w, h = gray.shape

    #croped = gray[int(w/2-32):int(w/2+32),int(h/2-32):int(h/2+32)]
    croped = gray[int(w*0.15):int(w*0.95),int(h*0.15):int(h*0.85)]
    croped = cv2.resize(croped, (64,64))
    cv2.imshow('',croped)
    cv2.waitKey(0)

'''
#faceCascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
#pimage = cv2.imread('./raw/ImposterRaw/0001/0001_00_00_01_0.jpg')
nima ge = cv2.imread('/raw/ClientRaw/0006/0006_00_00_01_169.jpg')
#crop = face.detect(pimage)
cv2.imshow('',nimage)
cv2.waitKey(0)

ncrop = face.detect(nimage)
cv2.imshow('',ncrop)
cv2.waitKey(0)

#cv2.imshow('face',crop)
#cv2.waitKey(0)

lbp = LBP()
#image_array = lbp.describe(crop)
#image_array = np.array(crop)
#lbp.show_revolve_hist(image_array)
nimage_array = np.array(ncrop)
lbp.show_revolve_hist(nimage_array)
#lbp.show_basic_hist(basic_array)
#lbp.show_image(basic_array)
'''