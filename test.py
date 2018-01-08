import cv2
import numpy as np
from lib import face
from lib.get_LBP_from_Image import LBP


#faceCascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
#pimage = cv2.imread('./raw/ImposterRaw/0001/0001_00_00_01_0.jpg')
nimage = cv2.imread('/raw/ClientRaw/0006/0006_00_00_01_169.jpg')
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
