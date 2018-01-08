import cv2
import numpy as np


def detect(image):
    # Input:
    #   image: original RGB image
    # Output:
    #   cropped image: cropped rgb image containing face

    faceCascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
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


def feat(lbp, img_path, method='revolve_uniform'):
    assert method in ['basic', 'revolve', 'uniform', 'revolve_uniform']
    img = cv2.imread(img_path)
    img = detect(img)
    if img is None:
        return None

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
        img_array = lbp.lbp_revolve_uniform(img)
        hist = lbp.get_revolve_uniform_hist(img_array)

    return hist