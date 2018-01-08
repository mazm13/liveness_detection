import os
import numpy as np
from sklearn import svm
from lib.utils import read_path
from lib.get_LBP_from_Image import LBP
from lib.face import feat

posi_test = 'client_test_raw.txt'
nega_test = 'imposter_test_raw.txt'


def load_training_data():
    posi = np.load('./datas/posi.npy')
    nega = np.load('./datas/nega.npy')
    data = np.concatenate((posi, nega))
    label = np.concatenate((np.ones(len(posi)), np.ones(len(nega) * -1)))
    return data, label


if __name__ == '__main__':
    method = 'revolve_uniform'
    data, label = load_training_data()
    clf = svm.SVC()
    clf.fit(data, label)

    # testing
    lbp = LBP()
    posi_paths = read_path(posi_test)
    nega_paths = read_path(nega_test)

    paths = posi_paths + nega_paths
    labels = [1] * len(posi_paths) + [-1] * len(nega_paths)

    total = len(labels)
    correct = 0
    for path, label in zip(paths, labels):
        feature = feat(lbp, path, method)
        if feature is None:
            pred = -1
        else:
            pred = clf.predict(feature)
        if pred == label:
            correct += 1

    print('Accuracy:{.2}'.format(float(correct) / total))
