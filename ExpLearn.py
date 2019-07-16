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
from Run_Racos import time_formulate
from ExpDataProcess import learning_data_load


class ImageNet(nn.Module):

    def __init__(self, middle_input_size=0, output_size=0):
        super(ImageNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, 2)
        self.conv2 = nn.Conv2d(4, 8, 2)
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(1, 2)
        self.fc1 = nn.Linear(64 + middle_input_size, 128)
        # self.dropout_linear1 = nn.Dropout2d(p=drop)
        self.fc2 = nn.Linear(128, 64)
        # self.dropout_linear2 = nn.Dropout2d(p=drop)
        self.fc3 = nn.Linear(64, output_size)
        # self.dropout_linear3 = nn.Dropout2d(p=drop)

    def forward(self, x):

        x2 = x[:, 0, x.size(2) - 1, :]
        x1 = x[:, :, 0:x.size(2) - 1, :]

        x1 = self.pool1(F.relu(self.conv1(x1)))
        x1 = self.pool2(F.relu(self.conv2(x1)))

        x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
        x = torch.cat((x1, x2), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x


class ComplexImageNet(nn.Module):

    def __init__(self, middle_input_size=0, output_size=0):
        super(ComplexImageNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, 2)
        self.conv2 = nn.Conv2d(4, 8, 2)
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(192 + middle_input_size, 128)
        # self.dropout_linear1 = nn.Dropout2d(p=drop)
        self.fc2 = nn.Linear(128, 64)
        # self.dropout_linear2 = nn.Dropout2d(p=drop)
        self.fc3 = nn.Linear(64, output_size)
        # self.dropout_linear3 = nn.Dropout2d(p=drop)

    def forward(self, x):

        x2 = x[:, 0, x.size(2) - 1, :]
        x1 = x[:, :, 0:x.size(2) - 1, :]

        x1 = self.pool1(F.relu(self.conv1(x1)))
        x1 = self.pool2(F.relu(self.conv2(x1)))

        # print x1

        x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
        x = torch.cat((x1, x2), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x


def exp_data_loader(file_path):

    f = open(file_path, 'rb')
    data = pickle.load(f)
    label = pickle.load(f)
    f.close()

    data_size = len(data)
    channel_size = len(data[0])
    length = len(data[0][0])
    width = len(data[0][0][0])
    label_size = len(label)

    print('data formulation: ', len(data), '*', len(data[0]), '*', len(data[0][0]), '*', len(data[0][0][0]))
    print('label size: ', len(label))

    data = [data, label]

    log_param = [data_size, channel_size, length, width, label_size]

    return data, log_param


def learning_data_transfer(instance_set=None):

    new_instance_set = []
    for i in range(len(instance_set)):
        new_instance_set.append([instance_set[i]])

    return new_instance_set


def split_data(data, batch_size=32, validation_rate=0.1):

    data, label = data

    data = batch_split(data, label, batch_size=batch_size)

    data_size = len(data)
    validation_size = int(data_size * validation_rate)

    train_data, validation_data = [], []

    for i in range(data_size):
        if i < validation_size:
            validation_data.append(data[i])
        else:
            train_data.append(data[i])

    print('--split data--')
    print('data size: ', data_size, ', train data size: ', len(train_data), ', validation data size: ',\
        len(validation_data))

    return train_data, validation_data


def batch_split(images, labels, batch_size=8):

        data = []
        each_batch_image = []
        each_batch_label = []
        for i in range(len(images)):

            each_batch_image.append(images[i])
            each_batch_label.append(labels[i])

            if (i + 1) % batch_size == 0:
                each_batch_image = torch.from_numpy(np.array(each_batch_image)).float()
                each_batch_label = torch.from_numpy(np.array(each_batch_label)).long()
                data.append([each_batch_image, each_batch_label])
                each_batch_image = []
                each_batch_label = []

        print('-------------------------------------')
        print('batch length: ', len(data))
        print('-------------------------------------')

        return data


def train_image_net():

    log_buffer = []

    random.seed(1)

    epoch_size = 100
    batch_size = 32
    vali_rate = 0.1
    learn_rate = 0.0005

    dim_size = 7
    categorical_size = 1

    validation_switch = True

    exp_up_name = 'mnist_svhn'
    exp_name = 'mnist_svhn2'
    total_path = './' + exp_up_name + '_exp/' + exp_name + '/'

    save_name = ''
    log_name = exp_name + '_lr' + str(learn_rate) + '_epoch' + str(epoch_size) + '_net_running_log.txt'
    # log_name = 'debug_running_log.xls'

    train_data_file = exp_name + '_exp_balance_data.pkl'
    test_data_file = exp_name + '_exp_training_data.pkl'

    train_data_name = total_path + train_data_file

    if test_data_file is not None:
        test_data_name = total_path + test_data_file
        print('loading testing data...')
        log_buffer.append('loading test data: ' + test_data_name)

        test_data, log_param = exp_data_loader(test_data_name)

        log_buffer.append('--' + 'data formulation: ' + str(log_param[0]) + '*' + str(log_param[1]) + '*' +
                          str(log_param[2]) + '*' + str(log_param[3]) + '--')
        log_buffer.append('--' + 'label size: ' + str(log_param[4]))
        print('loading testing data over.')
    else:
        test_data = None

    # train new models
    print('training model...---------------------------------------------------------------------------')
    print('loading training data...')
    log_buffer.append('loading train data: ' + train_data_name)

    data, log_param = exp_data_loader(train_data_name)

    log_buffer.append('--' + 'data formulation: ' + str(log_param[0]) + '*' + str(log_param[1]) + '*' +
                      str(log_param[2]) + '*' + str(log_param[3]) + '--')
    log_buffer.append('--' + 'label size: ' + str(log_param[4]))
    print('loading training data over.')

    print('split train and validation data...')
    trainloader, validationloader = split_data(data, batch_size=batch_size, validation_rate=vali_rate)
    log_buffer.append('--split data: train data size: ' + str(len(trainloader)) + ' validation data size: ' +
                      str(len(validationloader)))

    print('training net ================================================================')
    # data = mix_data(minority_data, majority_data[net_i])

    net_start = time.time()

    net = ImageNet(middle_input_size=dim_size, output_size=categorical_size)
    net.cuda()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)

    log_buffer.append('criterion: BCELoss, optimizer: Adam')
    log_buffer.append('--net train--')

    for epoch in range(epoch_size):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            labels = labels.float()             # BCELoss used

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                log_buffer.append('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

                # validation
                if validation_switch is True:
                    all_predictions, all_labels = [], []
                    for data in validationloader:
                        images, labels = data
                        images = Variable(images.cuda())

                        # print 'images:', images
                        # print 'labels:', labels

                        outputs = net(images)
                        outputs = outputs.cpu()

                        # _, predicted = torch.max(outputs.data, 1)

                        # for BCELoss
                        predicted = (torch.sign(outputs * 2 - 1) + 1) / 2
                        predicted = predicted.data.numpy()
                        predicted = predicted.reshape(predicted.size).astype(int).tolist()

                        all_predictions.extend(predicted)
                        all_labels.extend(labels.numpy().tolist())

                    # print all_predictions
                    # print all_labels

                    accuracy = accuracy_score(all_labels, all_predictions)
                    recall = recall_score(all_labels, all_predictions)
                    precision = precision_score(all_labels, all_predictions)

                    print('accuracy: ', accuracy, ', recall rate: ', recall, ', precision rate: ', precision)
                    log_buffer.append('accuracy: ' + str(accuracy) + ', recall rate: ' + str(recall) +
                                      ', precision rate: ' + str(precision))
    net_end = time.time()
    hour, minute, second = time_formulate(net_start, net_end)
    print('train net time: ', hour, ':', minute, ':', second)
    log_buffer.append('train net time: ' + str(hour) + ':' + str(minute) + ':' + str(second))

    test_start = time.time()
    if test_data is not None:
        print('testing=================================================')
        log_buffer.append('--net test--')

        testloader, _ = split_data(test_data, batch_size=batch_size, validation_rate=0.0)

        all_predictions, all_labels = [], []
        for data in testloader:
            images, labels = data
            images = Variable(images.cuda())

            # print 'images:', images
            # print 'labels:', labels

            outputs = net(images)
            outputs = outputs.cpu()

            # _, predicted = torch.max(outputs.data, 1)
            predicted = (torch.sign(outputs * 2 - 1) + 1) / 2
            predicted = predicted.data.numpy()
            predicted = predicted.reshape(predicted.size).astype(int).tolist()

            all_predictions.extend(predicted)
            all_labels.extend(labels.numpy().tolist())

        accuracy = accuracy_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)

        print('accuracy: ', accuracy, ', recall rate: ', recall, ', precision rate: ', precision)
        log_buffer.append('accuracy: ' + str(accuracy) + ', recall rate: ' + str(recall) +
                          ', precision rate: ' + str(precision))
    test_end = time.time()
    hour, minute, second = time_formulate(test_start, test_end)
    print('test net time: ', hour, ':', minute, ':', second)
    log_buffer.append('test net time: ' + str(hour) + ':' + str(minute) + ':' + str(second))

    net_file = total_path + exp_name + '_balance_net_lr' + str(learn_rate) + '_epoch' + str(epoch_size) + '.pkl'
    print('net saving...')
    torch.save(net, net_file)
    log_buffer.append('--net save: ' + net_file + '--')
    print('net saved!')

    log_file = total_path + log_name
    fo.FileWriter(log_file, log_buffer, style='w')

    return


def train_complex_image_net():

    log_buffer = []

    random.seed(1)

    epoch_size = 50
    batch_size = 32
    vali_rate = 0.1
    learn_rate = 0.0004

    dim_size = 19
    categorical_size = 1

    validation_switch = True

    exp_up_name = 'mnist'
    exp_name = 'complex_mnist'
    total_path = './' + exp_up_name + '_exp/' + exp_name + '/'

    save_name = ''
    log_name = exp_name + '_lr' + str(learn_rate) + '_epoch' + str(epoch_size) + '_net_running_log.txt'
    # log_name = 'debug_running_log.xls'

    train_data_file = exp_name + '_exp_balance_data.pkl'
    test_data_file = exp_name + '_exp_training_data.pkl'

    train_data_name = total_path + train_data_file

    if test_data_file is not None:
        test_data_name = total_path + test_data_file
        print('loading testing data...')
        log_buffer.append('loading test data: ' + test_data_name)

        test_data, log_param = exp_data_loader(test_data_name)

        log_buffer.append('--' + 'data formulation: ' + str(log_param[0]) + '*' + str(log_param[1]) + '*' +
                          str(log_param[2]) + '*' + str(log_param[3]) + '--')
        log_buffer.append('--' + 'label size: ' + str(log_param[4]))
        print('loading testing data over.')
    else:
        test_data = None

    # train new models
    print('training model...---------------------------------------------------------------------------')
    print('loading training data...')
    log_buffer.append('loading train data: ' + train_data_name)

    data, log_param = exp_data_loader(train_data_name)

    log_buffer.append('--' + 'data formulation: ' + str(log_param[0]) + '*' + str(log_param[1]) + '*' +
                      str(log_param[2]) + '*' + str(log_param[3]) + '--')
    log_buffer.append('--' + 'label size: ' + str(log_param[4]))
    print('loading training data over.')

    print('split train and validation data...')
    trainloader, validationloader = split_data(data, batch_size=batch_size, validation_rate=vali_rate)
    log_buffer.append('--split data: train data size: ' + str(len(trainloader)) + ' validation data size: ' +
                      str(len(validationloader)))

    print('training net ================================================================')
    # data = mix_data(minority_data, majority_data[net_i])

    net_start = time.time()

    net = ComplexImageNet(middle_input_size=dim_size, output_size=categorical_size)
    net.cuda()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)

    log_buffer.append('criterion: BCELoss, optimizer: Adam')
    log_buffer.append('--net train--')

    for epoch in range(epoch_size):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            labels = labels.float()             # BCELoss used

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                log_buffer.append('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

                # validation
                if validation_switch is True:
                    all_predictions, all_labels = [], []
                    for data in validationloader:
                        images, labels = data
                        images = Variable(images.cuda())

                        # print 'images:', images
                        # print 'labels:', labels

                        outputs = net(images)
                        outputs = outputs.cpu()

                        # _, predicted = torch.max(outputs.data, 1)

                        # for BCELoss
                        predicted = (torch.sign(outputs * 2 - 1) + 1) / 2
                        predicted = predicted.data.numpy()
                        predicted = predicted.reshape(predicted.size).astype(int).tolist()

                        all_predictions.extend(predicted)
                        all_labels.extend(labels.numpy().tolist())

                    # print all_predictions
                    # print all_labels

                    accuracy = accuracy_score(all_labels, all_predictions)
                    recall = recall_score(all_labels, all_predictions)
                    precision = precision_score(all_labels, all_predictions)

                    print('accuracy: ', accuracy, ', recall rate: ', recall, ', precision rate: ', precision)
                    log_buffer.append('accuracy: ' + str(accuracy) + ', recall rate: ' + str(recall) +
                                      ', precision rate: ' + str(precision))
    net_end = time.time()
    hour, minute, second = time_formulate(net_start, net_end)
    print('train net time: ', hour, ':', minute, ':', second)
    log_buffer.append('train net time: ' + str(hour) + ':' + str(minute) + ':' + str(second))

    test_start = time.time()
    if test_data is not None:
        print('testing=================================================')
        log_buffer.append('--net test--')

        testloader, _ = split_data(test_data, batch_size=batch_size, validation_rate=0.0)

        all_predictions, all_labels = [], []
        for data in testloader:
            images, labels = data
            images = Variable(images.cuda())

            # print 'images:', images
            # print 'labels:', labels

            outputs = net(images)
            outputs = outputs.cpu()

            # _, predicted = torch.max(outputs.data, 1)
            predicted = (torch.sign(outputs * 2 - 1) + 1) / 2
            predicted = predicted.data.numpy()
            predicted = predicted.reshape(predicted.size).astype(int).tolist()

            all_predictions.extend(predicted)
            all_labels.extend(labels.numpy().tolist())

        accuracy = accuracy_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)

        print('accuracy: ', accuracy, ', recall rate: ', recall, ', precision rate: ', precision)
        log_buffer.append('accuracy: ' + str(accuracy) + ', recall rate: ' + str(recall) +
                          ', precision rate: ' + str(precision))
    test_end = time.time()
    hour, minute, second = time_formulate(test_start, test_end)
    print('test net time: ', hour, ':', minute, ':', second)
    log_buffer.append('test net time: ' + str(hour) + ':' + str(minute) + ':' + str(second))

    net_file = total_path + exp_name + '_balance_net_lr' + str(learn_rate) + '_epoch' + str(epoch_size) + '.pkl'
    print('net saving...')
    torch.save(net, net_file)
    log_buffer.append('--net save: ' + net_file + '--')
    print('net saved!')

    log_file = total_path + log_name
    fo.FileWriter(log_file, log_buffer, style='w')

    return


def learning_exp():

    log_buffer = []

    random.seed(1)

    # training parameters
    epoch_size = 100
    batch_size = 32
    vali_rate = 0.1
    learn_rate = 0.0005
    categorical_size = 1
    validation_switch = True

    # exp data parameters
    dim_size = 10
    problem_name = 'sphere'
    start_index = 0
    bias_region = 0.2
    problem_num = 5

    learner_path = './ExpLearner/SyntheticProbsLearner/'
    data_path = './ExpLog/SyntheticProbsLog/'

    for prob_i in range(problem_num):

        log_buffer = []
        log_buffer.append('+++++++++++++++++++++++++++++++')
        log_buffer.append('training parameter')
        log_buffer.append('epoch size: ' + str(epoch_size))
        log_buffer.append('batch size: ' + str(batch_size))
        log_buffer.append('validation rate: ' + str(vali_rate))
        log_buffer.append('learning rate: ' + str(learn_rate))
        log_buffer.append('+++++++++++++++++++++++++++++++')
        log_buffer.append('experience data parameter')
        log_buffer.append('dimension size: ' + str(dim_size))
        log_buffer.append('problem name: ' + problem_name)
        log_buffer.append('problem index: ' + str(start_index + prob_i))
        log_buffer.append('+++++++++++++++++++++++++++++++')

        log_name = learner_path + problem_name + '/dimension' + str(dim_size) + '/TrainingLog/' + 'learning-log-' \
                   + problem_name + '-' + 'dim' + str(dim_size) + '-' + 'bias' + str(bias_region)\
                   + '-' + str(start_index + prob_i) + '.txt'

        data_file = data_path + problem_name + '/dimension' + str(dim_size) + '/LearningData/' + 'learning-data-' \
                    + problem_name + '-' + 'dim' + str(dim_size) + '-' + 'bias' + str(bias_region) + '-'\
                    + str(start_index + prob_i) + '.pkl'

        print('data loading: ', data_file)
        data_inf, bias, ori_data, blc_data = learning_data_load(file_path=data_file)

        train_data, train_label = blc_data
        test_data, test_label = ori_data

        print('data transfer...')
        train_data = learning_data_transfer(instance_set=train_data)
        test_data = learning_data_transfer(instance_set=test_data)

        print('test data formulation: ', len(test_data), '*', len(test_data[0]), '*', len(test_data[0][0]), '*',\
            len(test_data[0][0][0]))
        log_buffer.append('--' + 'test data formulation: ' + str(test_data) + '*' + str(test_data[0]) + '*'
                          + str(test_data[0][0]) + '*' + str(test_data[0][0][0]) + '--')
        print('test label size: ', len(test_label))
        log_buffer.append('--' + 'test label size: ' + str(len(test_label)))
        print('train data formulation: ', len(train_data), '*', len(train_data[0]), '*', len(train_data[0][0]), '*',\
            len(train_data[0][0][0]))
        log_buffer.append('--' + 'train data formulation: ' + str(train_data) + '*' + str(train_data[0]) + '*'
                          + str(train_data[0][0]) + '*' + str(train_data[0][0][0]) + '--')
        print('train label size: ', len(train_label))
        log_buffer.append('--' + 'train label size: ' + str(len(train_label)))

        # train new models
        print('training model...---------------------------------------------------------------------------')

        print('split train and validation data...')
        trainloader, validationloader = split_data([train_data, train_label], batch_size=batch_size,
                                                   validation_rate=vali_rate)
        log_buffer.append('--split data: train data size: ' + str(len(trainloader)) + ' validation data size: ' +
                          str(len(validationloader)))

        print('training net ================================================================')
        # data = mix_data(minority_data, majority_data[net_i])

        net_start = time.time()

        net = ImageNet(middle_input_size=dim_size, output_size=categorical_size)
        net.cuda()
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=learn_rate)

        log_buffer.append('criterion: BCELoss, optimizer: Adam')
        log_buffer.append('--net train--')

        for epoch in range(epoch_size):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                labels = labels.float()             # BCELoss used

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 50 == 49:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                    log_buffer.append('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

                    # validation
                    if validation_switch is True:
                        all_predictions, all_labels = [], []
                        for data in validationloader:
                            images, labels = data
                            images = Variable(images.cuda())

                            # print 'images:', images
                            # print 'labels:', labels

                            outputs = net(images)
                            outputs = outputs.cpu()

                            # _, predicted = torch.max(outputs.data, 1)

                            # for BCELoss
                            predicted = (torch.sign(outputs * 2 - 1) + 1) / 2
                            predicted = predicted.data.numpy()
                            predicted = predicted.reshape(predicted.size).astype(int).tolist()

                            all_predictions.extend(predicted)
                            all_labels.extend(labels.numpy().tolist())

                        # print all_predictions
                        # print all_labels

                        accuracy = accuracy_score(all_labels, all_predictions)
                        recall = recall_score(all_labels, all_predictions)
                        precision = precision_score(all_labels, all_predictions)

                        print('accuracy: ', accuracy, ', recall rate: ', recall, ', precision rate: ', precision)
                        log_buffer.append('accuracy: ' + str(accuracy) + ', recall rate: ' + str(recall) +
                                          ', precision rate: ' + str(precision))
        net_end = time.time()
        hour, minute, second = time_formulate(net_start, net_end)
        print('train net time: ', hour, ':', minute, ':', second)
        log_buffer.append('train net time: ' + str(hour) + ':' + str(minute) + ':' + str(second))

        test_start = time.time()
        if test_data is not None:
            print('testing=================================================')
            log_buffer.append('--net test--')

            testloader, _ = split_data([test_data, test_label], batch_size=batch_size, validation_rate=0.0)

            all_predictions, all_labels = [], []
            for data in testloader:
                images, labels = data
                images = Variable(images.cuda())

                # print 'images:', images
                # print 'labels:', labels

                outputs = net(images)
                outputs = outputs.cpu()

                # _, predicted = torch.max(outputs.data, 1)
                predicted = (torch.sign(outputs * 2 - 1) + 1) / 2
                predicted = predicted.data.numpy()
                predicted = predicted.reshape(predicted.size).astype(int).tolist()

                all_predictions.extend(predicted)
                all_labels.extend(labels.numpy().tolist())

            accuracy = accuracy_score(all_labels, all_predictions)
            recall = recall_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions)

            print('accuracy: ', accuracy, ', recall rate: ', recall, ', precision rate: ', precision)
            log_buffer.append('accuracy: ' + str(accuracy) + ', recall rate: ' + str(recall) +
                              ', precision rate: ' + str(precision))
        test_end = time.time()
        hour, minute, second = time_formulate(test_start, test_end)
        print('test net time: ', hour, ':', minute, ':', second)
        log_buffer.append('test net time: ' + str(hour) + ':' + str(minute) + ':' + str(second))

        net_file = learner_path + problem_name + '/dimension' + str(dim_size) + '/DirectionalModel/' + 'learner-' \
                   + problem_name + '-' + 'dim' + str(dim_size) + '-' + 'bias' + str(bias_region)\
                   + '-' + str(start_index + prob_i) + '.pkl'
        print('net saving...')
        torch.save(net, net_file)
        log_buffer.append('--net save: ' + net_file + '--')
        print('net saved!')

        fo.FileWriter(log_name, log_buffer, style='w')

    return


if __name__ == '__main__':

    # train_complex_image_net()
    learning_exp()