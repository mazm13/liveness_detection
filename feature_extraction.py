import numpy as np
from lib import face
from lib.utils import read_path

posi_train = 'client_train_raw.txt'
nega_train = 'imposter_train_raw.txt'

target = {'posi': 'client_train_raw.txt', 'nega': 'imposter_train_raw.txt'}

if __name__ == '__main__':

    method = 'uniform'

    for tar in target:
        paths = read_path(target[tar])
        datas = []
        miss_cnt = 0
        for path in paths:
            feature = face.feat(path, method)
            if feature is None:
                miss_cnt += 1
                continue
            else:
                datas.append(feature)
        datas = np.array(datas)
        np.save('./datas/' + tar + '_' + method, datas)
        print('there is(are) %d picture(s) cannot be detected' % miss_cnt)
