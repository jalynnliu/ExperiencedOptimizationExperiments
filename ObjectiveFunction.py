'''
Objective functions can be implemented in this file

Author:
    Yi-Qi Hu

Time:
    2016.6.13
'''

'''
 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

 Copyright (C) 2015 Nanjing University, Nanjing, China
'''

import math
from Tools import RandomOperator
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import classes
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import time
import pickle
import FileOperator as fo
import random
from Tools import list2string

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


# Sphere function for continue optimization
def Sphere(x):
    value = sum([(i - 0.2) * (i - 0.2) for i in x])
    return value


# Ackley function for continue optimization
def Ackley(x):
    bias = 0.2
    value_seq = 0
    value_cos = 0
    for i in range(len(x)):
        value_seq += (x[i] - bias) * (x[i] - bias)
        value_cos += math.cos(2.0 * math.pi * (x[i] - bias))
    ave_seq = value_seq / len(x)
    ave_cos = value_cos / len(x)
    value = -20 * math.exp(-0.2 * math.sqrt(ave_seq)) - math.exp(ave_cos) + 20.0 + math.e
    return value


# A test function for mixed optimization
def mixed_function(x):
    value = sum([i * i for i in x])
    return value


def three_types_function(x):
    xx = [i for i in x]
    for i in range(len(xx)):
        if i % 3 == 2:
            if xx[i] == 0:
                xx[i] = -50
            elif xx[i] == 1:
                xx[i] = 0
            else:
                xx[i] = 50
    value = sum([i * i for i in xx])
    return value


# set cover problem for discrete optimization
def SetCover(x):
    weight = [0.8356, 0.5495, 0.4444, 0.7269, 0.9960, 0.6633, 0.5062, 0.8429, 0.1293, 0.7355,
              0.7979, 0.2814, 0.7962, 0.1754, 0.0267, 0.9862, 0.1786, 0.5884, 0.6289, 0.3008]
    subset = []
    subset.append([0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0])
    subset.append([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0])
    subset.append([1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0])
    subset.append([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    subset.append([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    subset.append([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    subset.append([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0])
    subset.append([0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    subset.append([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0])
    subset.append([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
    subset.append([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0])
    subset.append([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1])
    subset.append([1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
    subset.append([1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1])
    subset.append([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
    subset.append([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0])
    subset.append([1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1])
    subset.append([0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1])
    subset.append([0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0])
    subset.append([0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1])

    allweight = 0
    countw = 0
    for i in range(len(weight)):
        allweight = allweight + weight[i]

    dims = []
    for i in range(len(subset[0])):
        dims.append(False)

    for i in range(len(subset)):
        if x[i] == 1:
            countw = countw + weight[i]
            for j in range(len(subset[i])):
                if subset[i][j] == 1:
                    dims[j] = True
    full = True
    for i in range(len(dims)):
        if dims[i] is False:
            full = False

    if full is False:
        countw = countw + allweight

    return countw


class DistributedFunction:

    def __init__(self, dim=None, bias_region=[-0.2, 0.2]):

        ro = RandomOperator()

        self.__dimension = dim
        # generate bias randomly
        self.__bias = []
        for i in range(self.__dimension.get_size()):
            # self.__bias.append(ro.getUniformDouble(self.__dimension.getRegion(i)[0], self.__dimension.getRegion(i)[1]))
            self.__bias.append(ro.get_uniform_double(bias_region[0], bias_region[1]))
        # print 'bias:', self.__bias

        return

    def getBias(self):
        return self.__bias

    def setBias(self, b):
        self.__bias = []
        for i in range(len(b)):
            self.__bias.append(b[i])
        # print 'bias:', self.__bias
        return

    def DisAckley(self, x):
        value_seq = 0
        value_cos = 0
        for i in range(len(x)):
            value_seq += (x[i] - self.__bias[i]) * (x[i] - self.__bias[i])
            value_cos += math.cos(2.0 * math.pi * (x[i] - self.__bias[i]))
        ave_seq = value_seq / len(x)
        ave_cos = value_cos / len(x)
        value = -20 * math.exp(-0.2 * math.sqrt(ave_seq)) - math.exp(ave_cos) + 20.0 + math.e
        return value

    def DisSphere(self, x):
        value = 0
        for i in range(len(x)):
            value += (x[i] - self.__bias[i]) * (x[i] - self.__bias[i])
        return value

    def DisRosenbrock(self, x):
        value = 0
        for i in range(len(x) - 1):
            xi = x[i] - self.__bias[i]
            value += (1 - xi) * (1 - xi) + 100 * (x[i + 1] - self.__bias[i] - xi * xi) * (
                        x[i + 1] - xi * xi - self.__bias[i])
        return value


# read data from files and return data on numpy.array type
def dataset_reader(train_file, test_file):
    f = open(train_file, 'rb')
    train_features = pickle.load(f)
    train_labels = pickle.load(f)
    f.close()

    f = open(test_file, 'rb')
    test_features = pickle.load(f)
    test_labels = pickle.load(f)
    f.close()

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    return train_features, train_labels, test_features, test_labels


class EnsembleClassifier:

    # X and Y are narray
    def __init__(self, X, Y, k=10):
        self.__X = X
        self.__Y = Y
        self.__feature_num = X.shape[1]

        self.__kf = StratifiedKFold(n_splits=k, shuffle=False)

        return

    def set_data(self, X, Y):
        self.__X = X
        self.__Y = Y
        return

    def get_data_info(self):
        return 'feature dimension:' + str(self.__X.shape[0]) + ', feature size:' + str(self.__X.shape[1])

    def data_collector(self, index_set):

        data = np.zeros((len(index_set), self.__feature_num))
        label = np.zeros(len(index_set))

        for i in range(index_set.shape[0]):
            data[i, :] = self.__X[index_set[i], :]
            label[i] = self.__Y[index_set[i]]
        return data, label

    def normalize(self, w):
        w = np.array(w)
        s = w.sum()
        w = (w / s).tolist()
        return w

    # x = [w0-w9, DTC_c, MLP_alpha, MLP_lr, MLP_h1, MLP_h2, MLP_h3, LR_c, SVC_c,
    #      PAC_c, SGDC_loss, SGDC_alpha, RFC_n, RFC_c, KNN_n]
    def get_accuracy(self, x):

        weight = self.normalize(x[0:10])

        if x[10] == 0:
            DTC_c = 'gini'
        else:
            DTC_c = 'entropy'

        MLP_alpha = x[11]
        MLP_lr = x[12]
        MLP_h = (x[13], x[14], x[15])

        LR_c = x[16]

        SVC_c = x[17]

        PAC_c = x[18]

        if x[19] == 0:
            SGDC_loss = 'hinge'
        elif x[19] == 1:
            SGDC_loss = 'log'
        elif x[19] == 2:
            SGDC_loss = 'modified_huber'
        elif x[19] == 3:
            SGDC_loss = 'squared_hinge'
        else:
            SGDC_loss = 'perceptron'
        SGDC_alpha = x[20]

        RFC_n = x[21]
        if x[22] == 0:
            RFC_c = 'gini'
        else:
            RFC_c = 'entropy'

        KNN_n = x[23]

        error_list = []

        fold_c = 0

        all_fold_start = time.time()

        for train_index, test_index in self.__kf.split(self.__X, self.__Y):
            # print 'fold ', fold_c+1, '-------------------------------------'
            fold_c += 1

            train_feature, train_label = self.data_collector(train_index)
            test_feature, test_label = self.data_collector(test_index)

            dtc = DecisionTreeClassifier(criterion=DTC_c)
            mlpc = MLPClassifier(hidden_layer_sizes=MLP_h, alpha=MLP_alpha, learning_rate_init=MLP_lr)
            lr = LogisticRegression(C=LR_c)
            svc = classes.SVC(C=SVC_c)
            gpc = GaussianProcessClassifier()
            pac = PassiveAggressiveClassifier(C=PAC_c)
            gnb = GaussianNB()
            sgdc = SGDClassifier(loss=SGDC_loss, alpha=SGDC_alpha)
            rfc = RandomForestClassifier(n_estimators=RFC_n, criterion=RFC_c)
            knn = KNeighborsClassifier(n_neighbors=KNN_n)

            estimators = [('dtc', dtc), ('mlpc', mlpc), ('lr', lr), ('svc', svc), ('gpc', gpc), ('pac', pac),
                          ('gnb', gnb), ('sgdc', sgdc), ('rfc', rfc), ('knn', knn)]

            voting = VotingClassifier(estimators, voting='hard', weights=weight, n_jobs=-1)

            each_fold_start = time.time()
            voting = voting.fit(train_feature, train_label)
            each_fold_end = time.time()

            predictions = voting.predict(test_feature)

            accuracy = accuracy_score(test_label, predictions)
            # print 'each fold error:', 1-accuracy, ', each fold train time:', time.ctime(each_fold_end - each_fold_start)

            error_list.append(1 - accuracy)

        all_fold_end = time.time()

        ave_error = np.mean(np.array(error_list))

        # print '-----------------------------------'
        # print 'average fold error:', ave_error, ', all folds train time:', time.ctime(all_fold_end - all_fold_start)

        return ave_error


class Net(nn.Module):
    def __init__(self, nn_structure, image_structure):
        super(Net, self).__init__()

        i_channel, i_length, i_width, i_categorical = image_structure
        con1_channel, con1_width, con2_channel, con2_width, den1_n, den2_n = nn_structure
        # cnn layers
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(i_channel, con1_channel, con1_width)
        self.conv2 = nn.Conv2d(con1_channel, con2_channel, con2_width)

        abs_length = (((i_length - con1_width + 1) / 2) - con2_width + 1) / 2
        abs_width = (((i_width - con1_width + 1) / 2) - con2_width + 1) / 2
        self.abs_feature_size = con2_channel * abs_length * abs_width
        # dense layer
        self.fc1 = nn.Linear(self.abs_feature_size, den1_n)
        self.fc2 = nn.Linear(den1_n, den2_n)
        self.fc3 = nn.Linear(den2_n, i_categorical)
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.abs_feature_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


def data_change(file_path):
    file_name = file_path + '_conv_old.txt'

    data = fo.FileReader(file_name)

    new_data = []
    for data_i in data:

        data_i = data_i.split(' ')

        new_data_i = []
        for each_num in data_i:
            each_num = float(each_num)
            each_num = each_num + random.normalvariate(0, 0.1)
            new_data_i.append(each_num)

        new_data.append(new_data_i)

    change_index = len(new_data) - 1

    for i in range(len(new_data[change_index]) - 1):
        if new_data[change_index][i] < new_data[change_index][i + 1]:
            new_data[change_index][i + 1] = new_data[change_index][i]

    print('print data: ', new_data[change_index])

    buff = []
    for buff_data in new_data:
        buff.append(list2string(buff_data))

    new_file_name = file_path + '_conv_change.txt'

    fo.FileWriter(new_file_name, buff, style='w')

    return


if __name__ == '__main__':
    # x = [6, 5, 16, 5, 120, 84, 0.001]

    # NN structure selection (nnss)
    # nnss = NNStructureSelection(dataset_name='MNIST')
    # validation_error = nnss.get_accuracy(x)
    #
    # print 'validation error: ', validation_error

    # r = [2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 3, 2, 2]
    # r = [1, 3, 1, 1, 3, 3, 1, 2, 1, 2, 2, 2, 3, 1, 3, 1, 1, 3, 1, 1, 4, 2, 1, 1]
    # r = [3, 5, 3, 1, 2, 1, 3, 3, 4, 3, 3, 1, 2, 3, 2, 3, 4, 2, 4, 4, 3, 1, 3, 5]
    # r = [6, 6, 6, 1, 4, 4, 4, 4, 6, 5, 5, 5, 6, 5, 4, 5, 5, 5, 5, 6, 6, 6, 4, 3]
    # r = [4, 4, 4, 6, 6, 5, 5, 5, 3, 6, 4, 4, 5, 4, 6, 4, 3, 4, 5, 5, 5, 5, 5, 4]
    # r = [5, 1, 5, 1, 5, 5, 6, 6, 5, 4, 6, 5, 4, 6, 4, 6, 6, 6, 1, 3, 2, 4, 5, 6]

    # r = [1, 3, 1, 1, 3, 2, 1, 1, 1, 1, 2, 1, 4, 1, 1, 2, 3, 2, 3, 1, 3, 3, 2, 1]
    # r = [1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 4, 2, 3, 2, 2, 2, 3, 1, 3, 3, 1, 1, 1, 2]
    # r = [3, 4, 6, 1, 4, 6, 3, 3, 4, 5, 1, 2, 1, 4, 6, 5, 1, 5, 2, 5, 2, 6, 4, 6]
    # r = [6, 5, 4, 1, 5, 4, 4, 6, 1, 3, 5, 2, 6, 5, 4, 6, 2, 3, 1, 6, 6, 4, 3, 4]
    # r = [3, 6, 5, 1, 2, 5, 6, 4, 5, 2, 5, 2, 2, 5, 5, 4, 3, 4, 3, 3, 3, 5, 6, 4]
    # r = [3, 1, 3, 1, 6, 3, 5, 4, 6, 6, 3, 2, 5, 3, 2, 1, 6, 6, 3, 2, 3, 1, 4, 2]

    # r = [1, 2, 2, 1, 1, 3, 2, 2, 1, 1]
    # r = [1, 1, 1, 2, 2, 1, 1, 1, 2, 2]
    # r = [1, 3, 3, 3, 3, 2, 3, 3, 4, 4]
    # r = [4, 5, 4, 4, 5, 5, 4, 5, 5, 5]
    # r = [5, 4, 5, 5, 5, 4, 5, 4, 6, 6]
    # r = [6, 6, 6, 6, 4, 6, 6, 6, 3, 2]

    # r = [2, 2, 1, 1, 3, 1, 1, 1, 2, 4]
    # r = [3, 2, 2, 1, 2, 3, 2, 1, 1, 1]
    # r = [4, 4, 3, 5, 1, 6, 3, 6, 3, 3]
    # r = [6, 6, 5, 3, 6, 4, 5, 4, 4, 6]
    # r = [5, 1, 4, 6, 4, 5, 6, 3, 5, 5]
    # r = [1, 5, 6, 4, 5, 2, 4, 4, 6, 2]

    # r = [1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
    # r = [2, 4, 2, 1, 2, 1, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 3, 2, 3, 3, 3, 1, 2, 4]
    # r = [5, 5, 5, 1, 3, 3, 3, 3, 5, 4, 4, 4, 5, 4, 3, 4, 4, 4, 4, 5, 5, 5, 3, 2]
    # r = [3, 3, 3, 5, 5, 4, 4, 4, 2, 5, 3, 3, 4, 3, 5, 3, 2, 3, 4, 4, 4, 4, 4, 3]
    # r = [4, 1, 4, 1, 4, 5, 5, 5, 4, 3, 5, 4, 3, 5, 3, 5, 5, 5, 1, 2, 2, 3, 4, 5]

    # r = [1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 3, 1, 1, 2, 3, 1, 3, 1, 2, 2, 1, 1]
    # r = [2, 3, 5, 1, 3, 5, 2, 2, 3, 4, 1, 2, 1, 3, 5, 4, 1, 4, 2, 4, 1, 5, 3, 5]
    # r = [5, 4, 3, 1, 4, 3, 3, 5, 1, 3, 4, 2, 5, 4, 3, 5, 2, 2, 1, 5, 5, 3, 2, 3]
    # r = [2, 5, 4, 1, 1, 4, 5, 3, 4, 2, 4, 2, 2, 4, 4, 3, 3, 3, 3, 3, 2, 4, 5, 3]
    # r = [2, 1, 2, 1, 5, 2, 4, 3, 5, 5, 3, 2, 4, 2, 2, 1, 5, 5, 3, 2, 2, 1, 3, 2]

    # r = [1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
    # r = [1, 2, 2, 2, 2, 1, 2, 2, 3, 3]
    # r = [3, 4, 3, 3, 4, 4, 3, 4, 4, 4]
    # r = [4, 3, 4, 4, 4, 3, 4, 3, 5, 5]
    # r = [5, 5, 5, 5, 3, 5, 5, 5, 2, 2]

    # r = [2, 2, 1, 1, 2, 1, 1, 1, 1, 3]
    # r = [3, 3, 2, 4, 1, 5, 2, 5, 2, 2]
    # r = [5, 5, 4, 2, 5, 3, 4, 3, 3, 5]
    # r = [4, 1, 3, 5, 3, 4, 5, 2, 4, 4]
    # r = [1, 4, 5, 3, 4, 2, 3, 3, 5, 1]
    # print len(r)
    #
    # print np.mean(np.array(r))

    file_total_path = './figure_data/'

    this_file = 'expracos_ackley_0.04_d10'

    data_change(file_total_path + this_file)
