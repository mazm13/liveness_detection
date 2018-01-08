import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq


def detect(image):
    # Input:
    #   image: original RGB image
    # Output:
    #   cropped image: cropped rgb image containing face

    faceCascade = cv2.CascadeClassifier("lib/haarcascade_frontalface_alt.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    #    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None

    # there may be more than one detected faces, select the biggest one
    face_scales = [w*h for (_, _, w, h) in faces]
    face_scales = np.array(face_scales)
    index = np.argmax(face_scales)
    (x, y, w, h) = faces[index]

    return gray[y: y+h, x: x+w]


def lbphist(img, method='default'):
    radius = 3
    no_points = 8 * radius
    lbp = local_binary_pattern(img, no_points, radius, method)
    x = itemfreq(lbp.ravel())
    hist = x[:, 1] / sum(x[:, 1])
    return hist


def feat(img_path, method='uniform'):
    assert method in ['default', 'ror', 'uniform', 'var']
    #print(img_path)
    image = cv2.imread(img_path)
    img = detect(image)
    if img is None:
        return None
    hist = lbphist(img, method=method)
    '''
    if method == 'basic':
        img_array = lbp.lbp_basic(img)
        hist = lbp.get_basic_hist(img_array)
    elif method == 'revolve':
        img_array = lbp.lbp_revolve(img)
        hist = lbp.get_revolve_hist(img_array)
    elif method == 'uniform':
        img_array = lbp.lbp_uniform(img)
        hist = lbp.get_uniform_hist(img_array)
    else:
        #img_array = lbp.lbp_revolve_uniform(img)
        img_array = img
        hist = lbp.get_revolve_uniform_hist(img_array)
    '''
    return hist
