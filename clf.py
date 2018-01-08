import os
import numpy as np
from sklearn import svm
from lib.utils import read_path
from lib.get_LBP_from_Image import LBP
from lib.face import feat

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
    method = 'uniform'
    data, label = load_training_data()
    #print(label.shape, data.shape)
    #print(label)
    #exit(0)
    clf = svm.SVC(C=0.1, gamma=2.0)
    clf.fit(data, label)

    # testing
    lbp = LBP()
    posi_paths = read_path(posi_test)
    nega_paths = read_path(nega_test)

    paths = posi_paths + nega_paths
    labels = [1] * len(posi_paths) + [-1] * len(nega_paths)

    total = len(labels)
    correct = 0
    ind = 0
    for path, label in zip(paths, labels):
        feature = feat(path, method)
        if feature is None:
            #print('None')
            pred = -1
        else:
            pred = clf.predict(np.expand_dims(feature,axis=0))
            #print(pred)
        if pred == label:
            correct += 1
        
        if ind % 10 == 0:
            print('[info]iter(%d,%d), pred=%d, label=%d' % (ind, total, pred, label))
        
        ind += 1

    print('Accuracy:{}'.format(float(correct) / total))
