
from matplotlib import pyplot as plt
import numpy as np


datas = np.load('./datas/posi_train.npy')
for i in range(10):
    data = datas[i]
    plt.plot(data, color='r')
plt.show()