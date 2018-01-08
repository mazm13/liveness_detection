import cv2
import numpy as np
from lib import face
from lib.get_LBP_from_Image import LBP
from matplotlib import pyplot as plt

import os
import numpy as np

posi_test = 'client_test_raw.txt'
nega_test = 'imposter_test_raw.txt'


def load_training_data():
    posi = np.load('./datas/posi_uniform.npy')
    nega = np.load('./datas/nega_uniform.npy')
    data = np.concatenate((posi, nega))
    label1 = np.ones(len(posi))
    label2 = np.ones(len(nega)) * -1
    #label = np.concatenate((np.ones(len(posi)), np.ones(len(nega)) * -1))
    label = np.concatenate((label1, label2))
    return data, label

if __name__ == '__main__':
    data, label = load_training_data()
    for i in range(len(data)):
        plt.plot(data[i], color='r')
        plt.xlim([0,60])
    plt.show()


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