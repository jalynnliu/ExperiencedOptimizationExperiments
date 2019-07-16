from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import FileOperator as fo
import numpy as np
import gc
import random
import xlwt
import time
import os
import pickle
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


class Net(nn.Module):

    def __init__(self, middle_input_size=0, output_size=0):
        super(Net, self).__init__()

        drop = 0.5

        # Net1
        if True:
            self.conv1 = nn.Conv2d(1, 6, 2)
            self.conv2 = nn.Conv2d(6, 16, 2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 + middle_input_size, 120)
            self.dropout_linear1 = nn.Dropout2d(p=drop)
            self.fc2 = nn.Linear(120, 84)
            self.dropout_linear2 = nn.Dropout(p=drop)
            self.fc3 = nn.Linear(84, output_size)
            self.dropout_linear3 = nn.Dropout(p=drop)

        # net 1.1 complex MLP layers
        if False:
            self.conv1 = nn.Conv2d(1, 6, 2)
            self.conv2 = nn.Conv2d(6, 16, 2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 + middle_input_size, 128)
            self.dropout_linear1 = nn.Dropout2d(p=drop)
            self.fc2 = nn.Linear(128, 64)
            self.dropout_linear2 = nn.Dropout(p=drop)
            self.fc3 = nn.Linear(64, 32)
            self.dropout_linear3 = nn.Dropout(p=drop)
            self.fc4 = nn.Linear(32, output_size)
            self.dropout_linear4 = nn.Dropout(p=drop)

        # Net2
        if False:

            self.conv1 = nn.Conv2d(1, 6, 2)
            self.conv2 = nn.Conv2d(6, 8, 2)
            self.conv3 = nn.Conv2d(8, 16, 2)
            self.conv4 = nn.Conv2d(16, 18, 2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(54 + middle_input_size, 120)
            self.dropout_linear1 = nn.Dropout2d(p=drop)
            self.fc2 = nn.Linear(120, 84)
            self.dropout_linear2 = nn.Dropout(p=drop)
            self.fc3 = nn.Linear(84, 36)
            self.dropout_linear3 = nn.Dropout(p=drop)
            self.fc4 = nn.Linear(36, output_size)
            self.dropout_linear4 = nn.Dropout(p=drop)

        # Net3
        if False:

            self.conv1 = nn.Conv2d(1, 6, (1, 2))
            self.pool = nn.AvgPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, (1, 2))
            self.fc1 = nn.Linear(80 + middle_input_size, 120)
            self.dropout_linear1 = nn.Dropout2d(p=drop)
            self.fc2 = nn.Linear(120, 84)
            self.dropout_linear2 = nn.Dropout(p=drop)
            self.fc3 = nn.Linear(84, output_size)
            self.dropout_linear3 = nn.Dropout(p=drop)

    def forward(self, x):
        x2 = x[:, 0, x.size(2) - 1, :]
        x1 = x[:, :, 0:x.size(2) - 1, :]

        # Net1
        if True:
            x1 = self.pool(F.relu(self.conv1(x1)))
            x1 = self.pool(F.relu(self.dropout_linear1(self.conv2(x1))))
            # x1 = self.pool(F.relu(self.conv1(x1)))
            # x1 = self.pool(F.relu(self.conv2(x1)))
            # print 'x1 matrix: ', x1
            x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
            # print 'x1: ', x1.size()
            x = torch.cat((x1, x2), -1)
            x = F.relu(self.dropout_linear2(self.fc1(x)))
            x = F.relu(self.dropout_linear3(self.fc2(x)))
            x = F.sigmoid(self.fc3(x))
            # x = F.sigmoid(self.fc3(x))

        # Net1.1
        if False:
            x1 = self.pool(F.relu(self.conv1(x1)))
            x1 = self.pool(F.relu(self.dropout_linear1(self.conv2(x1))))
            # x1 = self.pool(F.relu(self.conv1(x1)))
            # x1 = self.pool(F.relu(self.conv2(x1)))
            # print 'x1 matrix: ', x1
            x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
            # print 'x1: ', x1.size()
            x = torch.cat((x1, x2), -1)
            x = F.relu(self.dropout_linear2(self.fc1(x)))
            x = F.relu(self.dropout_linear3(self.fc2(x)))
            x = F.relu(self.dropout_linear4(self.fc3(x)))
            x = F.sigmoid(self.fc4(x))

        # Net2
        if False:
            x1 = self.pool(F.relu(self.conv2(F.relu(self.conv1(x1)))))
            x1 = self.pool(F.relu(self.dropout_linear1(self.conv4(F.relu(self.conv3(x1))))))
            x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
            x = torch.cat((x1, x2), -1)
            x = F.relu(self.dropout_linear2(self.fc1(x)))
            x = F.relu(self.dropout_linear3(self.fc2(x)))
            x = F.relu(self.dropout_linear4(self.fc3(x)))
            x = F.sigmoid(self.fc4(x))

        # Net3
        if False:
            x1 = self.pool(F.relu(self.conv1(x1)))
            x1 = self.pool(F.relu(self.dropout_linear1(self.conv2(x1))))
            # print 'x1 matrix: ', x1
            x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
            # print 'x1: ', x1.size()
            x = torch.cat((x1, x2), -1)
            x = F.relu(self.dropout_linear2(self.fc1(x)))
            x = F.relu(self.dropout_linear3(self.fc2(x)))
            x = F.sigmoid(self.fc3(x))

        return x


class EC_Net(nn.Module):

    def __init__(self, middle_input_size=0, output_size=0):
        super(EC_Net, self).__init__()

        drop = 0.5

        # Net1
        if True:
            self.conv1 = nn.Conv2d(1, 6, 2)
            self.conv2 = nn.Conv2d(6, 16, 2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(320 + middle_input_size, 512)
            self.dropout_linear1 = nn.Dropout2d(p=drop)
            self.fc2 = nn.Linear(512, 64)
            self.dropout_linear2 = nn.Dropout2d(p=drop)
            self.fc3 = nn.Linear(64, output_size)
            self.dropout_linear3 = nn.Dropout2d(p=drop)

        # net 1.1 complex MLP layers
        if False:
            self.conv1 = nn.Conv2d(1, 6, 2)
            self.conv2 = nn.Conv2d(6, 16, 2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(320 + middle_input_size, 128)
            self.dropout_linear1 = nn.Dropout2d(p=drop)
            self.fc2 = nn.Linear(128, 64)
            self.dropout_linear2 = nn.Dropout(p=drop)
            self.fc3 = nn.Linear(64, 32)
            self.dropout_linear3 = nn.Dropout(p=drop)
            self.fc4 = nn.Linear(32, output_size)
            self.dropout_linear4 = nn.Dropout(p=drop)

        # Net2
        if False:

            self.conv1 = nn.Conv2d(1, 6, 2)
            self.conv2 = nn.Conv2d(6, 8, 2)
            self.conv3 = nn.Conv2d(8, 16, 2)
            self.conv4 = nn.Conv2d(16, 18, 2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(54 + middle_input_size, 120)
            self.dropout_linear1 = nn.Dropout2d(p=drop)
            self.fc2 = nn.Linear(120, 84)
            self.dropout_linear2 = nn.Dropout(p=drop)
            self.fc3 = nn.Linear(84, 36)
            self.dropout_linear3 = nn.Dropout(p=drop)
            self.fc4 = nn.Linear(36, output_size)
            self.dropout_linear4 = nn.Dropout(p=drop)

        # Net3
        if False:

            self.conv1 = nn.Conv2d(1, 6, (1, 2))
            self.pool = nn.AvgPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, (1, 2))
            self.fc1 = nn.Linear(80 + middle_input_size, 120)
            self.dropout_linear1 = nn.Dropout2d(p=drop)
            self.fc2 = nn.Linear(120, 84)
            self.dropout_linear2 = nn.Dropout(p=drop)
            self.fc3 = nn.Linear(84, output_size)
            self.dropout_linear3 = nn.Dropout(p=drop)

    def forward(self, x):
        x2 = x[:, 0, x.size(2) - 1, :]
        x1 = x[:, :, 0:x.size(2) - 1, :]

        # Net1
        if True:
            x1 = self.pool(F.relu(self.conv1(x1)))
            x1 = self.pool(F.relu(self.dropout_linear1(self.conv2(x1))))
            # x1 = self.pool(F.relu(self.conv1(x1)))
            # x1 = self.pool(F.relu(self.conv2(x1)))
            # print 'x1 matrix: ', x1
            x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
            # print 'x1: ', x1.size()
            x = torch.cat((x1, x2), -1)
            x = F.relu(self.dropout_linear2(self.fc1(x)))
            x = F.relu(self.dropout_linear3(self.fc2(x)))
            x = F.sigmoid(self.fc3(x))
            # x = F.sigmoid(self.fc3(x))

        # Net1.1
        if False:
            x1 = self.pool(F.relu(self.conv1(x1)))
            x1 = self.pool(F.relu(self.dropout_linear1(self.conv2(x1))))
            # x1 = self.pool(F.relu(self.conv1(x1)))
            # x1 = self.pool(F.relu(self.conv2(x1)))
            # print 'x1 matrix: ', x1
            x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
            # print 'x1: ', x1.size()
            x = torch.cat((x1, x2), -1)
            x = F.relu(self.dropout_linear2(self.fc1(x)))
            x = F.relu(self.dropout_linear3(self.fc2(x)))
            x = F.relu(self.dropout_linear4(self.fc3(x)))
            x = F.sigmoid(self.fc4(x))

        # Net2
        if False:
            x1 = self.pool(F.relu(self.conv2(F.relu(self.conv1(x1)))))
            x1 = self.pool(F.relu(self.dropout_linear1(self.conv4(F.relu(self.conv3(x1))))))
            x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
            x = torch.cat((x1, x2), -1)
            x = F.relu(self.dropout_linear2(self.fc1(x)))
            x = F.relu(self.dropout_linear3(self.fc2(x)))
            x = F.relu(self.dropout_linear4(self.fc3(x)))
            x = F.sigmoid(self.fc4(x))

        # Net3
        if False:
            x1 = self.pool(F.relu(self.conv1(x1)))
            x1 = self.pool(F.relu(self.dropout_linear1(self.conv2(x1))))
            # print 'x1 matrix: ', x1
            x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
            # print 'x1: ', x1.size()
            x = torch.cat((x1, x2), -1)
            x = F.relu(self.dropout_linear2(self.fc1(x)))
            x = F.relu(self.dropout_linear3(self.fc2(x)))
            x = F.sigmoid(self.fc3(x))

        return x


class my_predictor:

    def __init__(self, nets_path=''):
        self.__nets = self.load_predictors(path=nets_path)
        return

    def load_predictors(self, path=''):

        print ''
        # print 'load models ****************************************************'
        buff = fo.FileReader(path + 'nets_setting.txt')
        net_len = int(buff[0])
        # print 'number of nets: ', net_len
        file_name_list = []
        for i in xrange(net_len):
            t_name = buff[i + 1]
            t_name = t_name.strip('\n')
            file_name_list.append(t_name)
        net_list = []
        for i in xrange(net_len):
            model = torch.load(path + file_name_list[i])
            net_list.append(model)
            # print file_name_list[i], ' loaded!'
        # print '****************************************************************'
        return net_list

    def do_prediction(self, x):

        a = []
        a.append(x)
        b = []
        b.append(a)
        x = np.array(b)
        x = torch.from_numpy(x)
        # print x.size()
        x = Variable(x.cuda())

        predictions = []
        for i in xrange(len(self.__nets)):
            net = self.__nets[i]
            output = net(x)
            predictions.append(output[0, 0])

        mean_pre = np.mean(np.array(predictions))

        return mean_pre


# function inputs are features and label file names
# function output is a list [features (tunnel*height*width), label]
# params is a list of data informations
#       params name, trajectory size, tunnel size, width, height, label size
def load_data(x_file='', label_file=''):

    params = []
    params.append('load_data')

    # get x
    str_data = fo.FileReader(x_file)
    data_len = 0
    i = 0
    tensor_1D = []
    tensor_2D = []
    tensor_3D = []
    tensor_4D = []
    while i < len(str_data):
        # if i % 10000 == 0:
        #     print 'iteration ', i
        if str_data[i] == ' \n':
            data_len += 1
            tensor_3D.append(tensor_2D)
            tensor_4D.append(tensor_3D)
            tensor_2D = []
            tensor_3D = []
            while i < len(str_data) and str_data[i] == ' \n':
                i += 1
        else:
            str_nums = str_data[i].split(' ')
            tensor_1D = []
            for j in xrange(len(str_nums)):
                tensor_1D.append(float(str_nums[j]))
            tensor_2D.append(tensor_1D)
            i += 1
    print 'data set information---------------------------------------------------'
    print 'transform of trajectory successful!'
    print 'trajectory size:', len(tensor_4D)
    params.append(len(tensor_4D))
    print 'tunnel size:', len(tensor_4D[0])
    params.append(len(tensor_4D[0]))
    print 'sample size:', len(tensor_4D[0][0]), '*', len(tensor_4D[0][0][0])
    params.append(len(tensor_4D[0][0]))
    params.append(len(tensor_4D[0][0][0]))
    del str_data, tensor_1D, tensor_2D, tensor_3D,  str_nums
    gc.collect()

    # get labels
    str_labels = fo.FileReader(label_file)
    labels = []
    labels_tensor = []
    for i in xrange(len(str_labels)):
        t_label = float(str_labels[i])
        if t_label == -1.0:
            t_label = 0.0
        labels_tensor.append(t_label)
    print 'label size:', len(labels_tensor)
    params.append(len(labels_tensor))

    # data = []
    # data_temp = []
    # label_temp = []
    # for i in xrange(len(tensor_4D)):
    #     data_temp.append(tensor_4D[i])
    #     label_temp.append(labels_tensor[i])
    #     if (i+1) % batch_size == 0:
    #         # data.append([np.array(data_temp, dtype='float32'), np.array(label_temp)])
    #         data.append([np.array(data_temp), np.array(label_temp)])
    #         data_temp = []
    #         label_temp = []
    # print 'mini-batch size:', len(data)
    print '-----------------------------------------------------------------------'
    data = [tensor_4D, labels_tensor]
    return data, params


# params: training data size, validation data size
def split_data(data, validation_rate=0.0):

    params = []
    params.append('split_data')

    print 'training and validation data splitting---------------------------------'
    features, labels = data
    if len(features) != len(labels):
        print 'data length error!'
        exit(0)
    data_len = len(features)
    validation_len = int(data_len * validation_rate)
    train_len = data_len - validation_len

    split_index = [i for i in xrange(data_len)]
    random.shuffle(split_index)
    # print 'split date random: ', split_index[0:10]

    train_features = []
    train_labels = []
    for i in xrange(train_len):
        train_features.append(features[split_index[i]])
        train_labels.append(labels[split_index[i]])
    train_data = [train_features, train_labels]
    print 'training data size: ', len(train_labels)
    params.append(len(train_labels))

    validation_features = []
    validation_labels = []
    for i in xrange(train_len, data_len):
        validation_features.append(features[split_index[i]])
        validation_labels.append(labels[split_index[i]])
    validation_data = [validation_features, validation_labels]
    print 'validation data size: ', len(validation_labels)
    params.append(len(validation_labels))

    del features, labels
    gc.collect()
    print '-----------------------------------------------------------------------'
    return train_data, validation_data, params


# for unbalanced binary classification problem, find which label is minority and split majority into several parts
# return minority data set and a list of majority data set, params: minority label, majority label, minority size,
#  majority size, number of sub-sets, sub-set1 size,..., sub-sets_m size
def split_minority_data(data):

    params = []
    params.append('split_minority_data')

    features, labels = data
    pos_features = []
    pos_labels = []
    neg_features = []
    neg_labels = []
    pos_num = 0
    neg_num = 0
    print 'split minority data----------------------------------------------'
    for i in xrange(len(labels)):
        if labels[i] == 1:
            pos_num += 1
            pos_features.append(features[i])
            pos_labels.append(labels[i])
        else:
            neg_num += 1
            neg_features.append(features[i])
            neg_labels.append(labels[i])
    del features, labels, data
    gc.collect()
    if pos_num < neg_num:
        minority_label = 1
        minority_num = pos_num
        minority_features = pos_features
        minority_labels = pos_labels
        majority_label = 0
        majority_num = neg_num
        majority_features = neg_features
        majority_labels = neg_labels
    else:
        minority_label = 0
        minority_num = neg_num
        minority_features = neg_features
        minority_labels = neg_labels
        majority_label = 1
        majority_num = pos_num
        majority_features = pos_features
        majority_labels = pos_labels
    minority_data = [minority_features, minority_labels]
    subset_num = int(round(float(majority_num)/minority_num))
    if subset_num % 2 == 0:
        subset_num -= 1
    print 'minority: label is ', minority_label, ', size is ', minority_num, '|majority: label is ', majority_label, ', size is ', majority_num, '|subset size: ', subset_num
    params.append(minority_label)
    params.append(majority_label)
    params.append(minority_num)
    params.append(majority_num)
    params.append(subset_num)
    each_subset = int(majority_num/subset_num)
    split_index = [i for i in xrange(majority_num)]
    random.shuffle(split_index)
    # print 'split majority data: ', split_index[0:10]
    subset = []
    sub_majority_features = []
    sub_majority_labels = []
    i = 0
    j = 0
    while True:
        sub_majority_features.append(majority_features[split_index[i]])
        sub_majority_labels.append(majority_labels[split_index[i]])
        i += 1
        if i % each_subset == 0:
            subset.append([sub_majority_features, sub_majority_labels])
            j += 1
            print 'sub-set ', j, ': data size is ', len(sub_majority_labels)
            params.append(len(sub_majority_labels))
            sub_majority_features = []
            sub_majority_labels = []
            if j == subset_num-1:
                break
    while i < majority_num:
        sub_majority_features.append(majority_features[split_index[i]])
        sub_majority_labels.append(majority_labels[split_index[i]])
        i += 1
    subset.append([sub_majority_features, sub_majority_labels])
    print 'sub-set ', j+1, ': data size is ', len(sub_majority_labels)
    params.append(len(sub_majority_labels))
    return minority_data, subset, params


#
def mix_data(data1, data2):

    features1, labels1 = data1
    features2, labels2 = data2

    temp_features = []
    temp_labels = []
    temp_features.extend(features1)
    temp_features.extend(features2)
    temp_labels.extend(labels1)
    temp_labels.extend(labels2)

    index = [i for i in xrange(len(temp_labels))]
    random.shuffle(index)
    # print 'mix data random: ', index[0:10]
    features = []
    labels = []
    for i in xrange(len(index)):
        features.append(temp_features[index[i]])
        labels.append(temp_labels[index[i]])

    data = [features, labels]
    return data


# the input data should be constructed by feature and label,
# the feature is  3D list (tunnels*height*width) and label is an number of 1 or 0
# data = [[feature, label]; [feature, label]; ...; [feature, label]]
# the batch size should be larger than 1
# the output is a list constructed by batch of features and labels
# output = [[4D numpy array (batch size*tunnels*height*width), 1D numpy array (batch size*1)]; ...]
# params: batch size, number of batches
def construct_training_data(data, batch_size=1):

    params = []
    params.append('construct_training_data')

    features, labels = data
    temp_data = []
    temp_label = []
    new_data = []
    for i in xrange(len(labels)):
        temp_data.append(features[i])
        temp_label.append(labels[i])
        if (i+1) % batch_size == 0:
            new_data.append([np.array(temp_data), np.array(temp_label)])
            temp_data = []
            temp_label = []
    print 'construct training format-------------------------------------------'
    print 'batch size: ', batch_size, ' | number of batches: ', len(new_data)
    params.append(batch_size)
    params.append(len(new_data))
    print 'number of data before constructing: ', len(labels)
    print '--------------------------------------------------------------------'
    del features, labels
    gc.collect()
    return new_data, params


# return params list includes all parameters generating in training net
def train_net(train_data=None, test_data=None, epoch_size=1, batch_size=1, validation_rate=0.0, middle_input_size=1, output_size=1, train_file=''):

    params_list = []

    train_data, validation_data, params = split_data(train_data, validation_rate=validation_rate)
    params_list.append(params)
    train_data, params = construct_training_data(train_data, batch_size=batch_size)
    params_list.append(params)
    validation_data, params = construct_training_data(validation_data, batch_size=batch_size)
    params_list.append(params)
    net = EC_Net(middle_input_size=middle_input_size, output_size=output_size)
    net.double()
    net.cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # training
    train_params = []
    train_params.append('train_net')
    train_params.append(train_file)
    print ''
    print 'begin training....'

    # learn_rate = [0.1, 0.02, 0.01,  0.002, 0.001, 0.0002, 0.0001, 0.00002, 0.00001, 0.000002]
    for epoch in xrange(epoch_size):

        print ''
        print 'epoch ', epoch, '----------------------------------------------------------'
        # print 'learning rate: ', lr_set

        running_loss = 0.0

        for i, data in enumerate(train_data, 0):

            inputs, labels = data
            # numpy to torch.Tensor
            inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
            # Tensor to Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            labels = labels.double()

            optimizer.zero_grad()

            outputs = net(inputs)
            # print 'training outputs:', outputs.view(-1, outputs.size(0))
            # print 'labels'
            # print labels
            # print 'outputs: ', outputs
            # print 'labels: ', labels
            loss = criterion(outputs, labels)
            # print 'each loss:', loss
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            base_num = 1000
            if i % base_num == (base_num-1) or i == (len(train_data)-1):
                if i != len(train_data)-1:
                    train_loss = running_loss / base_num
                else:
                    train_loss = running_loss / (len(train_data)%base_num)
                print '[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, train_loss)

                params = []
                params.append(epoch+1)
                params.append(i+1)
                params.append(train_loss)

                running_loss = 0.0

                correct = 0
                total = 0
                total_loss = 0.0
                num_true = 0
                num_pos = 0
                TP = 0
                for vali_data in validation_data:
                    inputs, labels = vali_data
                    inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    labels = labels.double()
                    outputs = net(inputs)
                    # print '------testing outputs: ', outputs.view(-1, outputs.size(0))
                    predictions = (torch.sign(outputs * 2 - 1) + 1) / 2
                    # print '------predictions: ', predictions.view(-1, predictions.size(0))
                    # print '------target     : ', labels.view(-1, labels.size(0))
                    TP += ((predictions * 2 - 1) == labels).sum().data[0]
                    loss = criterion(outputs, labels)
                    total_loss += loss.data[0]
                    num_true += labels.sum().data[0]
                    num_pos += predictions.sum().data[0]
                    total += 1
                    correct += (predictions == labels).sum().data[0]

                # print 'total loss:'
                # print total_loss
                # print 'correct:'
                # print correct
                # print 'total:'
                # print total
                if num_pos != 0:
                    pre_rate = float(TP) / num_pos
                else:
                    pre_rate = 0.0
                if num_true != 0:
                    recall_rate = float(TP) / num_true
                else:
                    recall_rate = 0.0
                print 'validation loss: %.3f  |validation accuracy: %.3f %%' % (
                total_loss / total, 100 * correct / (total * batch_size))
                params.append(total_loss / total)
                params.append(float(correct) / (total * batch_size))
                print 'precision rate : %.3f %%|recall rate        : %.3f %%' % (
                100 * pre_rate, 100 * recall_rate)
                params.append(pre_rate)
                params.append(recall_rate)
                train_params.append(params)
    params_list.append(train_params)
    print 'training is over.'
    # training over
    # testing model
    print ''
    if test_data is not None:
        print 'begin testing##################################################################################'
        test_data, params = construct_training_data(test_data, batch_size=batch_size)
        params_list.append(params)
        correct = 0
        total = 0
        total_loss = 0.0
        num_true = 0
        num_pos = 0
        TP = 0
        for i, data in enumerate(test_data):
            inputs, labels = data
            inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net(inputs)
            predictions = (torch.sign(outputs * 2 - 1) + 1) / 2
            # print 'training prediction:', predictions
            TP += ((predictions * 2 - 1) == labels).sum().data[0]
            loss = criterion(outputs, labels)
            total_loss += loss.data[0]
            num_true += labels.sum().data[0]
            num_pos += predictions.sum().data[0]
            total += 1
            correct += (predictions == labels).sum().data[0]
        if num_pos != 0:
            pre_rate = float(TP) / num_pos
        else:
            pre_rate = 0.0
        if num_true != 0:
            recall_rate = float(TP) / num_true
        else:
            recall_rate = 0.0
        params = []
        params.append('testing')
        print 'test loss:       %.3f |test accuracy: %.3f %%' % (total_loss / total, 100 * correct / (total * batch_size))
        print 'precision rate : %.3f %%|recall rate  : %.3f %%' % (
            100 * pre_rate, 100 * recall_rate)
        params.append(total_loss / total)
        params.append(correct / (total * batch_size))
        params.append(pre_rate)
        params.append(recall_rate)
    params_list.append(params)
    return net, params_list


# ensemble predictor
# if the number of predictors which give positive label is the same as the ones which give negative label,
# the label will be given randomly
# return to list, first is ensemble label, the second is 2d list, each row is label of each net
def voting(net_list=None, data=None, batch_size=1):

    print 'classifiers size: ', len(net_list)

    test_data, params = construct_training_data(data, batch_size=batch_size)

    all_predictions = []
    for n_i in xrange(len(net_list)):
        count_net_half_one = 0
        this_outputs = []
        for f_i, data in enumerate(test_data):
            feature, label = data
            inputs = torch.from_numpy(feature)
            inputs = Variable(inputs.cuda())
            outputs = net_list[n_i](inputs)
            # predictions = (torch.sign(outputs * 2 - 1) + 1) / 2
            # predictions = predictions.data.cpu().numpy()
            # all_predictions.extend(predictions.reshape(predictions.size).tolist)
            outputs = outputs.data.cpu().numpy()
            outputs = outputs.reshape(outputs.size).tolist()
            this_outputs.extend(outputs)
        all_predictions.append(this_outputs)
    predictions = np.array(all_predictions)
    predictions = np.mean(predictions, axis=0)
    predictions = torch.from_numpy(predictions)
    predictions = (torch.sign(predictions * 2 - 1) + 1) / 2
    predictions = predictions.numpy().tolist()

    all_predictions = torch.from_numpy(np.array(all_predictions))
    all_predictions = (torch.sign(all_predictions * 2 - 1) + 1) / 2
    all_predictions = all_predictions.numpy().tolist()

    # label_matrix = (label_matrix + 1) / 2
    # predictions in [-1,1], label_matrix in [-1,0,1]
    return predictions, all_predictions


def save_nets(net_list, path='', net_name='net'):

    print ''
    print 'save models $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
    buff = []
    t_str = str(len(net_list))
    buff.append(t_str)
    file_name_list = []
    for i in xrange(len(net_list)):
        t_str = net_name + str(i) + '.pkl'
        file_name_list.append(t_str)
        buff.append(t_str)
    fo.FileWriter(path+net_name+'_setting.txt', buff)
    for i in xrange(len(file_name_list)):
        file_name = path + file_name_list[i]
        torch.save(net_list[i], file_name)
        print file_name_list[i], ' saved!'
    print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'


def load_nets(path=''):

    print ''
    print 'load models ****************************************************'
    buff = fo.FileReader(path + 'nets_setting.txt')
    net_len = int(buff[0])
    print 'number of nets: ', net_len
    file_name_list = []
    for i in xrange(net_len):
        t_name = buff[i+1]
        t_name = t_name.strip('\n')
        file_name_list.append(t_name)
    net_list = []
    for i in xrange(net_len):
        model = torch.load(path + file_name_list[i])
        net_list.append(model)
        print file_name_list[i], ' loaded!'
    print '****************************************************************'
    return net_list


# comparing each prediction generated by each predictor and ensemble result with ground true
# the number of predictors is m, the results is a list of m+1 lines
def ensemble_testing(predictions, label_matrix, ground_truth, batch_size, test_file):

    ens_params = []
    ens_params.append('ensemble_testing')
    ens_params.append(test_file)

    len_ground = len(ground_truth)
    ground_truth = np.array(ground_truth[0:(len_ground - (len_ground % batch_size))])

    len_predictors = label_matrix.size(0)
    len_labels = label_matrix.size(1)
    results = []
    # each predictor result
    for i in xrange(len_predictors):
        this_prediction = label_matrix[i]
        accuracy = accuracy_score(ground_truth, this_prediction)
        precision_rate = precision_score(ground_truth, this_prediction)
        recall_rate = recall_score(ground_truth, this_prediction)
        this_result = []
        params = [i+1]
        this_result.append(accuracy)
        this_result.append(precision_rate)
        this_result.append(recall_rate)
        results.append(this_result)
        params.extend(this_result)
        ens_params.append(params)
    # ensemble result
    accuracy = accuracy_score(ground_truth, predictions)
    precision_rate = precision_score(ground_truth, predictions)
    recall_rate = recall_score(predictions, ground_truth)
    this_result = []
    params = ['ensemble']
    this_result.append(accuracy)
    this_result.append(precision_rate)
    this_result.append(recall_rate)
    results.append(this_result)
    params.extend(this_result)
    ens_params.append(params)

    return results, ens_params


# comparing prediction with ground truth
# return accuracy, precision rate and recall rate
def do_testing(predictions, labels):

    correct = (predictions == labels).sum()
    # print 'correct num: ', correct
    accuracy = float(correct) / predictions.size(1)
    # print 'accuracy:', accuracy
    TP = (((predictions * 2) - 1) == labels).sum()
    # print 'TP:', TP
    num_true = labels.sum()
    # print 'number of true:', num_true
    # for i in xrange(predictions.size(1)):
    #     if predictions[0, i] != 1.0 and predictions[0, i] != 0.0:
    #         print i, ':', predictions[0, i]
    num_pos = predictions.sum()
    # print 'number of positive:', num_pos
    # print 'ensemble test accuracy: %.3f %%' % (100 * accuracy)
    # print 'precision rate: %.3f %% | recall rate: %.3f %%' % (100 * float(TP) / num_pos, 100 * float(TP) / num_true)

    if num_true == 0:
        recall_rate = 0
    else:
        recall_rate = float(TP) / num_true

    if num_pos == 0:
        precision_rate = 0
    else:
        precision_rate = float(TP) / num_pos

    return accuracy, precision_rate, recall_rate


def format_time(start_t, end_t, t_type):
    t = end_t - start_t
    hour_t = int(t / 3600)
    t -= (hour_t*3600)
    minute_t = int(t / 60)
    t -= (minute_t * 60)
    second_t = t
    params = []
    params.append('time')
    params.append(t_type)
    params.append(hour_t)
    params.append(minute_t)
    params.append(second_t)
    return params


def write_excel(xls_buff, path='training_log.xls'):

    load_data_title = ['trajectory size', 'tunnel size', 'width', 'height', 'label size']
    split_min_data_title = ['minority label', 'majority label', 'minority size', 'majority size', 'number of subset']
    split_data_title = ['training data size', 'validation data size']
    construct_train_data_title = ['batch size', 'number of batches']
    train_net_title = ['epoch index', 'batch index', 'training loss', 'validation loss', 'validation accuracy',
                      'precision rate', 'recall rate']
    testing_title = ['testing loss', 'testing accuracy', 'precision rate', 'recall rate']
    ensemble_testing_title = ['predictor index', 'accuracy', 'precision rate', 'recall rate']
    time_title = ['type', 'hours', 'minutes', 'seconds']

    xls_index = 0

    workbook = xlwt.Workbook()

    data_sheet = workbook.add_sheet('sheet1', cell_overwrite_ok=True)

    for index, params in enumerate(xls_buff):

        # print 'params 0:', params[0]

        if params[0] == 'load_data':    # tested
            data_sheet.write(xls_index, 0, 'load_data')
            for i in xrange(len(load_data_title)):
                data_sheet.write(xls_index, i+1, load_data_title[i])
            xls_index += 1
            for i in xrange(len(params)-1):
                data_sheet.write(xls_index, i+1, params[i+1])
        elif params[0] == 'split_minority_data':
            data_sheet.write(xls_index, 0, 'split_minority_data')
            for i in xrange(len(params)-1):
                if i < 5:
                    data_sheet.write(xls_index, i+1, split_min_data_title[i])
                    data_sheet.write(xls_index+1, i+1, params[i+1])
                else:
                    data_sheet.write(xls_index, i+1, 'sub-set '+str(i-4)+' size')
                    data_sheet.write(xls_index+1, i+1, params[i+1])
            xls_index += 1
        elif params[0] == 'split_data':
            data_sheet.write(xls_index, 0, 'split_data')
            for i in xrange(len(split_data_title)):
                data_sheet.write(xls_index, i+1, split_data_title[i])
            xls_index += 1
            for i in xrange(len(params)-1):
                data_sheet.write(xls_index, i+1, params[i+1])
        elif params[0] == 'construct_training_data':
            data_sheet.write(xls_index, 0, 'construct_training_data')
            for i in xrange(len(construct_train_data_title)):
                data_sheet.write(xls_index, i+1, construct_train_data_title[i])
            xls_index += 1
            for i in xrange(len(params)-1):
                data_sheet.write(xls_index, i+1, params[i+1])
        elif params[0] == 'train_net':
            data_sheet.write(xls_index, 0, 'training_net')
            data_sheet.write(xls_index+1, 0, params[1])
            for i in xrange(len(train_net_title)):
                data_sheet.write(xls_index, i+1, train_net_title[i])
            for i in xrange(len(params)-2):
                xls_index += 1
                for j in xrange(len(params[i+2])):
                    data_sheet.write(xls_index, j+1, params[i+2][j])
        elif params[0] == 'testing':
            data_sheet.write(xls_index, 0, 'testing_in_training')
            for i in xrange(len(testing_title)):
                data_sheet.write(xls_index, i+1, testing_title[i])
            xls_index += 1
            for i in xrange(len(params)-1):
                data_sheet.write(xls_index, i+1, params[i+1])
        elif params[0] == 'ensemble_testing':  # tested
            data_sheet.write(xls_index, 0, 'ensemble_testing')
            data_sheet.write(xls_index+1, 0, params[1])
            for i in xrange(len(ensemble_testing_title)):
                data_sheet.write(xls_index, i+1, ensemble_testing_title[i])
            for i in xrange(len(params)-2):
                xls_index += 1
                for j in xrange(len(params[i+2])):
                    data_sheet.write(xls_index, j+1, params[i+2][j])
        elif params[0] == 'time':   # tested
            data_sheet.write(xls_index, 0, 'time')
            for i in xrange(len(time_title)):
                data_sheet.write(xls_index, i+1, time_title[i])
            xls_index += 1
            for i in xrange(len(params)-1):
                data_sheet.write(xls_index, i+1, params[i+1])
        else:
            print 'params[0]: ', params[0]
        xls_index += 2

    workbook.save(path)
    print 'running log saved!'
    return


def train_nn_for_function():
    xls_buff = []

    random.seed(1)

    epoch_size = 20
    batch_size = 32
    vali_rate = 0.1

    dim_size = 10

    total_path = 'trained_CNN_models/ec_model1/'

    save_name = ''
    net_name = 'func_ee_net'
    log_name = 'lr0.01_dropout0.5_net1.1_running_log.xls'
    # log_name = 'debug_running_log.xls'

    train_x = 'top_8000_1_train_trajectory.txt'
    train_label = 'top_8000_1_train_label.txt'
    test_x = 'top_8000_1_test_trajectory.txt'
    test_label = 'top_8000_1_test_label.txt'
    # train_x = 'debug_trajectory.txt'
    # train_label = 'debug_label.txt'
    # test_x = 'debug_trajectory.txt'
    # test_label = 'debug_label.txt'

    train_x_file = total_path + train_x
    train_label_file = total_path + train_label
    test_x_file = total_path + test_x
    test_label_file = total_path + test_label
    save_path = total_path + save_name
    log_path = total_path + log_name

    # train new models
    if True:
        print 'training model...---------------------------------------------------------------------------'
        print 'loading training data...'
        data, params = load_data(x_file=train_x_file, label_file=train_label_file)
        xls_buff.append(params)
        print 'loading training data over.'
        print 'loading testing data...'
        test_data, params = load_data(x_file=test_x_file, label_file=test_label_file)
        xls_buff.append(params)
        print 'splitting minority data...'
        minority_data, majority_data, params = split_minority_data(data)
        xls_buff.append(params)
        del data
        gc.collect()
        majority_set_num = len(majority_data)
        net_list = []

        ensemble_start = time.time()

        for net_i in xrange(3):  # majority_set_num
            print 'training net ', net_i, '================================================================'
            data = mix_data(minority_data, majority_data[net_i])
            net_start = time.time()
            net, params_list = train_net(train_data=data, test_data=test_data, epoch_size=epoch_size,
                                         batch_size=batch_size,
                                         validation_rate=vali_rate, middle_input_size=dim_size, output_size=1,
                                         train_file=train_x_file)
            net_end = time.time()
            net_list.append(net)
            xls_buff.extend(params_list)
            params = format_time(net_start, net_end, 'train a net')
            xls_buff.append(params)

        save_nets(net_list=net_list, path=save_path, net_name=net_name)

        ensemble_end = time.time()
        params = format_time(ensemble_start, ensemble_end, 'train ensemble learner')
        xls_buff.append(params)

    # load trained models
    if False:
        print 'loading testing data...'
        test_data, params = load_data(x_file=test_x_file, label_file=test_label_file)
        xls_buff.append(params)
        net_list = load_nets(path=save_path)

    print ''
    print 'ensemble testing***************************************************************'

    ens_test_start = time.time()

    predictions, label_matrix = voting(net_list=net_list, data=test_data, batch_size=batch_size)
    labels = test_data[1]
    # print 'debug labels: ', labels
    # labels = torch.from_numpy(np.array(labels))

    results, params = ensemble_testing(predictions, label_matrix, labels, batch_size, test_x_file)
    xls_buff.append(params)

    ens_test_end = time.time()

    params = format_time(ens_test_start, ens_test_end, 'ensemble test')
    xls_buff.append(params)

    print 'predictor | accuracy | precision rate | recall rate'
    print '---------------------------------------------------'
    for i in xrange(len(results) - 1):
        this_result = results[i]
        print '    %2d    |  %.2f%%  |     %.2f%%     |   %.2f%%' % (
        (i + 1), 100 * this_result[0], 100 * this_result[1], 100 * this_result[2])
    this_result = results[len(results) - 1]
    print ' ensemble |  %.2f%%  |     %.2f%%     |   %.2f%%' % (
    100 * this_result[0], 100 * this_result[1], 100 * this_result[2])
    print '---------------------------------------------------'

    print 'logging...'
    write_excel(xls_buff, log_path)
    # predictions = predictions.double()
    # # print 'predictions: ', predictions
    # labels = test_data[1]
    # len_labels = len(labels)
    # labels = np.array(labels[0:(len_labels-(len_labels % batch_size))])
    # labels = torch.from_numpy(labels)
    # # labels = Variable(labels)
    # # print 'true labels: ', labels
    # correct = (predictions == labels).sum()
    # # print 'correct num: ', correct
    # accuracy = float(correct)/predictions.size(1)
    # # print 'accuracy:', accuracy
    # TP = (((predictions * 2) - 1) == labels).sum()
    # # print 'TP:', TP
    # num_true = labels.sum()
    # # print 'number of true:', num_true
    # for i in xrange(predictions.size(1)):
    #     if predictions[0, i] != 1.0 and predictions[0, i] != 0.0:
    #         print i, ':', predictions[0, i]
    # num_pos = predictions.sum()
    # print 'number of positive:', num_pos
    # print 'ensemble test accuracy: %.3f %%' % (100 * accuracy)
    # print 'precision rate: %.3f %% | recall rate: %.3f %%' % (100*float(TP)/num_true, 100*float(TP)/num_pos)
    print '-------'


# procedure for ec training
def ec_data_loader(file_name):

    params = []
    params.append('load_data')

    f = open(file_name, 'rb')
    data = pickle.load(f)
    label = pickle.load(f)
    f.close()

    params.append(len(data))
    params.append(len(data[0]))
    params.append(len(data[0][0]))
    params.append(len(data[0][0][0]))
    params.append(len(label))

    data = [data, label]

    return data, params


def train_net_for_ec(train_data=None, test_data=None, epoch_size=1, batch_size=1, validation_rate=0.0, middle_input_size=1, output_size=1, train_file=''):

    params_list = []

    train_data, validation_data, params = split_data(train_data, validation_rate=validation_rate)
    params_list.append(params)
    train_data, params = construct_training_data(train_data, batch_size=batch_size)
    params_list.append(params)
    validation_data, params = construct_training_data(validation_data, batch_size=batch_size)
    params_list.append(params)
    net = EC_Net(middle_input_size=middle_input_size, output_size=output_size)
    net.double()
    net.cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # training
    train_params = []
    train_params.append('train_net')
    train_params.append(train_file)
    print ''
    print 'begin training....'

    # learn_rate = [0.1, 0.02, 0.01,  0.002, 0.001, 0.0002, 0.0001, 0.00002, 0.00001, 0.000002]
    for epoch in xrange(epoch_size):

        print ''
        print 'epoch ', epoch, '----------------------------------------------------------'
        # print 'learning rate: ', lr_set

        running_loss = 0.0

        for i, data in enumerate(train_data, 0):

            inputs, labels = data
            # numpy to torch.Tensor
            inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
            # Tensor to Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            labels = labels.double()

            optimizer.zero_grad()

            outputs = net(inputs)
            # print 'training outputs:', outputs.view(-1, outputs.size(0))
            # print 'labels'
            # print labels
            # print 'outputs: ', outputs
            # print 'labels: ', labels
            loss = criterion(outputs, labels)
            # print 'each loss:', loss
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            base_num = 100
            if i % base_num == (base_num-1) or i == (len(train_data)-1):
                if i != len(train_data)-1:
                    train_loss = running_loss / base_num
                else:
                    train_loss = running_loss / (len(train_data)%base_num)
                print '[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, train_loss)

                params = []
                params.append(epoch+1)
                params.append(i+1)
                params.append(train_loss)

                running_loss = 0.0

                all_labels = []
                all_predictions = []
                total_loss = 0.0
                total = 0
                for vali_data in validation_data:
                    inputs, labels = vali_data
                    all_labels.extend(labels.tolist())
                    inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    labels = labels.double()
                    outputs = net(inputs)
                    # print '------testing outputs: ', outputs.view(-1, outputs.size(0))
                    predictions = (torch.sign(outputs * 2 - 1) + 1) / 2
                    predictions = predictions.data.cpu().numpy()
                    predictions = predictions.reshape(predictions.size)
                    all_predictions.extend(predictions.tolist())
                    # print '------predictions: ', predictions.view(-1, predictions.size(0))
                    # print '------target     : ', labels.view(-1, labels.size(0))
                    # TP += ((predictions * 2 - 1) == labels).sum().data[0]
                    loss = criterion(outputs, labels)
                    total_loss += loss.data[0]
                    # num_true += labels.sum().data[0]
                    # num_pos += predictions.sum().data[0]
                    total += 1
                    # correct += (predictions == labels).sum().data[0]

                # print 'total loss:'
                # print total_loss
                # print 'correct:'
                # print correct
                # print 'total:'
                # print total
                # if num_pos != 0:
                #     pre_rate = float(TP) / num_pos
                # else:
                #     pre_rate = 0.0
                # if num_true != 0:
                #     recall_rate = float(TP) / num_true
                # else:
                #     recall_rate = 0.0
                accuracy = accuracy_score(all_labels, all_predictions)
                recall = recall_score(all_labels, all_predictions)
                precision = precision_score(all_labels, all_predictions)
                print 'validation loss: %.3f  |validation accuracy: %.3f %%' % (
                total_loss / total, 100*accuracy)
                params.append(total_loss / total)
                params.append(accuracy)
                print 'precision rate : %.3f %%|recall rate        : %.3f %%' % (
                100 * precision, 100 * recall)
                params.append(precision)
                params.append(recall)
                train_params.append(params)
    params_list.append(train_params)
    print 'training is over.'
    # training over
    # testing model
    print ''
    if test_data is not None:
        print 'begin testing##################################################################################'
        test_data, params = construct_training_data(test_data, batch_size=batch_size)
        params_list.append(params)
        total = 0
        total_loss = 0.0
        all_labels = []
        all_predictions = []
        for i, data in enumerate(test_data):
            inputs, labels = data
            all_labels.extend(labels.tolist())
            inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            labels = labels.double()
            outputs = net(inputs)
            predictions = (torch.sign(outputs * 2 - 1) + 1) / 2
            predictions = predictions.data.cpu().numpy()
            predictions = predictions.reshape(predictions.size)
            all_predictions.extend(predictions.tolist())
            # print 'training prediction:', predictions
            loss = criterion(outputs, labels)
            total_loss += loss.data[0]
            total += 1
        params = []
        params.append('testing')
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        print 'test loss:       %.3f |test accuracy: %.3f %%' % (total_loss / total, 100 * accuracy)
        print 'precision rate : %.3f %%|recall rate  : %.3f %%' % (100 * precision, 100 * recall)
        params.append(total_loss / total)
        params.append(accuracy)
        params.append(precision)
        params.append(recall)
    params_list.append(params)
    return net, params_list


def train_nn_for_ec():
    xls_buff = []

    random.seed(1)

    epoch_size = 40
    batch_size = 32
    vali_rate = 0.1

    dim_size = 24

    total_path = 'trained_CNN_models/ec_model1/'

    save_name = ''
    net_name = 'vs_net'
    log_name = 'lr0.001_dropout0.5_net1_running_log.xls'
    # log_name = 'debug_running_log.xls'

    train_data_file = 'exp_model_balance_data.pkl'
    test_data_file = None
    # train_x = 'debug_trajectory.txt'
    # train_label = 'debug_label.txt'
    # test_x = 'debug_trajectory.txt'
    # test_label = 'debug_label.txt'

    train_data_name = total_path + train_data_file
    save_path = total_path + save_name
    log_path = total_path + log_name

    if test_data_file is not None:
        test_data_name = total_path + test_data_file
        print 'loading testing data...'
        test_data, params = ec_data_loader(test_data_name)
        xls_buff.append(params)
        print 'loading testing data over.'
    else:
        test_data is None


    # train new models
    if True:
        print 'training model...---------------------------------------------------------------------------'
        print 'loading training data...'
        data, params = ec_data_loader(train_data_name)
        xls_buff.append(params)
        print 'loading training data over.'
        # print 'loading testing data...'
        # test_data, params = load_data(x_file=test_x_file, label_file=test_label_file)
        # xls_buff.append(params)
        # print 'splitting minority data...'
        # minority_data, majority_data, params = split_minority_data(data)
        # xls_buff.append(params)
        # del data
        # gc.collect()
        # majority_set_num = len(majority_data)
        net_list = []

        ensemble_start = time.time()
        print 'training net ================================================================'
        # data = mix_data(minority_data, majority_data[net_i])
        net_start = time.time()
        net, params_list = train_net_for_ec(train_data=data, test_data=test_data, epoch_size=epoch_size,
                                            batch_size=batch_size,
                                            validation_rate=vali_rate, middle_input_size=dim_size, output_size=1,
                                            train_file=train_data_file)
        net_end = time.time()
        net_list.append(net)
        xls_buff.extend(params_list)
        params = format_time(net_start, net_end, 'train a net')
        xls_buff.append(params)

        save_nets(net_list=net_list, path=save_path, net_name=net_name)

        ensemble_end = time.time()
        params = format_time(ensemble_start, ensemble_end, 'train ensemble learner')
        xls_buff.append(params)

    # load trained models
    # if False:
    #     print 'loading testing data...'
    #     test_data, params = load_data(x_file=test_x_file, label_file=test_label_file)
    #     xls_buff.append(params)
    #     net_list = load_nets(path=save_path)

    if test_data is not None:
        print ''
        print 'testing***************************************************************'
        ens_test_start = time.time()

        predictions, all_predictions = voting(net_list=net_list, data=test_data, batch_size=batch_size)
        labels = test_data[1]
        # print 'debug labels: ', labels
        # labels = torch.from_numpy(np.array(labels))

        results, params = ensemble_testing(predictions, all_predictions, labels, batch_size, test_data_file)
        xls_buff.append(params)

        ens_test_end = time.time()

        params = format_time(ens_test_start, ens_test_end, 'ensemble test')
        xls_buff.append(params)

        print 'predictor | accuracy | precision rate | recall rate'
        print '---------------------------------------------------'
        for i in xrange(len(results) - 1):
            this_result = results[i]
            print '    %2d    |  %.2f%%  |     %.2f%%     |   %.2f%%' % (
                (i + 1), 100 * this_result[0], 100 * this_result[1], 100 * this_result[2])
        this_result = results[len(results) - 1]
        print ' ensemble |  %.2f%%  |     %.2f%%     |   %.2f%%' % (
            100 * this_result[0], 100 * this_result[1], 100 * this_result[2])
        print '---------------------------------------------------'

    # ens_test_start = time.time()
    #
    # predictions, label_matrix = voting(net_list=net_list, data=test_data, batch_size=batch_size)
    # labels = test_data[1]
    # # print 'debug labels: ', labels
    # # labels = torch.from_numpy(np.array(labels))
    #
    # results, params = ensemble_testing(predictions, label_matrix, labels, batch_size, test_x_file)
    # xls_buff.append(params)
    #
    # ens_test_end = time.time()
    #
    # params = format_time(ens_test_start, ens_test_end, 'ensemble test')
    # xls_buff.append(params)
    #
    # print 'predictor | accuracy | precision rate | recall rate'
    # print '---------------------------------------------------'
    # for i in xrange(len(results) - 1):
    #     this_result = results[i]
    #     print '    %2d    |  %.2f%%  |     %.2f%%     |   %.2f%%' % (
    #     (i + 1), 100 * this_result[0], 100 * this_result[1], 100 * this_result[2])
    # this_result = results[len(results) - 1]
    # print ' ensemble |  %.2f%%  |     %.2f%%     |   %.2f%%' % (
    # 100 * this_result[0], 100 * this_result[1], 100 * this_result[2])
    # print '---------------------------------------------------'
    #
    print 'logging...'
    write_excel(xls_buff, log_path)


# train nn_model by easy ensemble framework
def train_ee_nn_for_ec():
    xls_buff = []

    random.seed(1)

    epoch_size = 40
    batch_size = 32
    vali_rate = 0.1

    dim_size = 24

    total_path = 'trained_CNN_models/ec_model1/ee_net/'

    save_name = ''
    net_name = 'ee_net'
    log_name = 'lr0.001_dropout0.5_net1_running_log.xls'
    # log_name = 'debug_running_log.xls'

    train_data_files = ['exp_model_ee_data0.pkl', 'exp_model_ee_data1.pkl', 'exp_model_ee_data2.pkl',
                        'exp_model_ee_data3.pkl', 'exp_model_ee_data4.pkl']
    test_data_file = 'exp_model_train_data.pkl'
    # train_x = 'debug_trajectory.txt'
    # train_label = 'debug_label.txt'
    # test_x = 'debug_trajectory.txt'
    # test_label = 'debug_label.txt'

    save_path = total_path + save_name
    log_path = total_path + log_name

    if test_data_file is not None:
        test_data_name = total_path + test_data_file
        print 'loading testing data...'
        test_data, params = ec_data_loader(test_data_name)
        xls_buff.append(params)
        print 'loading testing data over.'
        print 'test data info:'
        print 'data size: ', len(test_data[0]), ', positive feature size: ', sum(np.array(test_data[1]))
    else:
        test_data = None

    # train new models
    if True:
        ensemble_start = time.time()
        print 'training model...---------------------------------------------------------------------------'
        nets = []
        for net_i in xrange(len(train_data_files)):
            print 'training net ', net_i, '----------------------------'
            train_data_file = train_data_files[net_i]
            train_data_name = total_path + train_data_file
            print 'loading training data...'
            data, params = ec_data_loader(train_data_name)
            xls_buff.append(params)
            print 'loading training data over.'
            print 'training data info:'
            print 'data size: ', len(data[0]), ', positive feature size: ', sum(np.array(data[1]))

            net_list = []

            ensemble_start = time.time()
            print 'training net ================================================================'
            # data = mix_data(minority_data, majority_data[net_i])
            net_start = time.time()
            net, params_list = train_net_for_ec(train_data=data, test_data=test_data, epoch_size=epoch_size,
                                                batch_size=batch_size,
                                                validation_rate=vali_rate, middle_input_size=dim_size, output_size=1,
                                                train_file=train_data_file)
            net_end = time.time()
            nets.append(net)
            xls_buff.extend(params_list)
            params = format_time(net_start, net_end, 'train a net')
            xls_buff.append(params)

        save_nets(net_list=nets, path=save_path, net_name=net_name)

        ensemble_end = time.time()
        params = format_time(ensemble_start, ensemble_end, 'train ensemble learner')
        xls_buff.append(params)

    # load trained models
    # if False:
    #     print 'loading testing data...'
    #     test_data, params = load_data(x_file=test_x_file, label_file=test_label_file)
    #     xls_buff.append(params)
    #     net_list = load_nets(path=save_path)

    if test_data is not None:
        print ''
        print 'ensemble testing***************************************************************'

        ens_test_start = time.time()

        predictions, all_predictions = voting(net_list=net_list, data=test_data, batch_size=batch_size)
        labels = test_data[1]
        # print 'debug labels: ', labels
        # labels = torch.from_numpy(np.array(labels))

        results, params = ensemble_testing(predictions, all_predictions, labels, batch_size, test_data_file)
        xls_buff.append(params)

        ens_test_end = time.time()

        params = format_time(ens_test_start, ens_test_end, 'ensemble test')
        xls_buff.append(params)

        print 'predictor | accuracy | precision rate | recall rate'
        print '---------------------------------------------------'
        for i in xrange(len(results) - 1):
            this_result = results[i]
            print '    %2d    |  %.2f%%  |     %.2f%%     |   %.2f%%' % (
            (i + 1), 100 * this_result[0], 100 * this_result[1], 100 * this_result[2])
        this_result = results[len(results) - 1]
        print ' ensemble |  %.2f%%  |     %.2f%%     |   %.2f%%' % (
        100 * this_result[0], 100 * this_result[1], 100 * this_result[2])
        print '---------------------------------------------------'

    print 'logging...'

    write_excel(xls_buff, log_path)
    # predictions = predictions.double()
    # # print 'predictions: ', predictions
    # labels = test_data[1]
    # len_labels = len(labels)
    # labels = np.array(labels[0:(len_labels-(len_labels % batch_size))])
    # labels = torch.from_numpy(labels)
    # # labels = Variable(labels)
    # # print 'true labels: ', labels
    # correct = (predictions == labels).sum()
    # # print 'correct num: ', correct
    # accuracy = float(correct)/predictions.size(1)
    # # print 'accuracy:', accuracy
    # TP = (((predictions * 2) - 1) == labels).sum()
    # # print 'TP:', TP
    # num_true = labels.sum()
    # # print 'number of true:', num_true
    # for i in xrange(predictions.size(1)):
    #     if predictions[0, i] != 1.0 and predictions[0, i] != 0.0:
    #         print i, ':', predictions[0, i]
    # num_pos = predictions.sum()
    # print 'number of positive:', num_pos
    # print 'ensemble test accuracy: %.3f %%' % (100 * accuracy)
    # print 'precision rate: %.3f %% | recall rate: %.3f %%' % (100*float(TP)/num_true, 100*float(TP)/num_pos)
    print '-------'


if __name__ == '__main__':

    train_ee_nn_for_ec()



