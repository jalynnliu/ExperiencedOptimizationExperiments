from Racos import RacosOptimization
from ObjectiveFunction import DistributedFunction
from Components import Dimension
from Tools import list2string
import time
import FileOperator as fo
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import random
import pickle
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from Run_Racos import time_formulate
from ExpDataProcess import learning_data_load, learning_instance_construct, learning_instance_balance
from ExpLearn import learning_data_transfer, split_data, batch_split
from SyntheticProbsSample import read_log, save_log
import copy
from Tools import string2list
from ExpRacos import ExpRacosOptimization

path = '/data/ExpAdaptation'
sample_size = 10  # the instance number of sampling in an iteration
budget = 500  # budget in online style
positive_num = 2  # the set size of PosPop
rand_probability = 0.99  # the probability of sample in model
uncertain_bits = 20  # the dimension size that is sampled randomly

start_index = 511
problem_name = 'sphere'
problem_num = 2000 - start_index

repeat_num = 10

exp_path = path + '/ExpLog/SyntheticProbsLog/'

bias_region = 0.5

dimension_size = 100

dimension = Dimension()
dimension.set_dimension_size(dimension_size)
dimension.set_regions([[-1.0, 1.0] for _ in range(dimension_size)], [0 for _ in range(dimension_size)])



class ImageNet(nn.Module):

    def __init__(self, middle_input_size=0, output_size=0):
        super(ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (1, 10))
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, (1, 10))
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, (1, 10))
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, (1, 10))
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(1, 2)
        self.fc1 = nn.Linear(2560 + middle_input_size, 256)
        # self.dropout_linear1 = nn.Dropout2d(p=drop)
        self.fc2 = nn.Linear(256, 64)
        # self.dropout_linear2 = nn.Dropout2d(p=drop)
        self.fc3 = nn.Linear(64, output_size)
        # self.dropout_linear3 = nn.Dropout2d(p=drop)

    def forward(self, x):
        x2 = x[:, 0, x.size(2) - 1, :]
        x1 = x[:, :, 0:x.size(2) - 1, :]
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = self.pool1(x1)
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = self.pool2(x1)

        x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))
        x = torch.cat((x1, x2), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x

# experience sample
def synthetic_problems_sample():
    for prob_i in range(problem_num):

        # bias log format: 'index,bias_list: dim1 dim2 dim3...'
        bias_log = []
        running_log = []
        running_log.append('+++++++++++++++++++++++++++++++++')
        running_log.append('optimization setting: ')
        running_log.append('sample_size: ' + str(sample_size))
        running_log.append('positive_num: ' + str(positive_num))
        running_log.append('rand_probability: ' + str(rand_probability))
        running_log.append('uncertain_bits: ' + str(uncertain_bits))
        running_log.append('budget: ' + str(budget))
        running_log.append('+++++++++++++++++++++++++++++++++')

        print(problem_name, ': ', start_index + prob_i, ' ==============================================')
        running_log.append(
            problem_name + ': ' + str(start_index + prob_i) + ' ==============================================')

        # problem setting
        func = DistributedFunction(dim=dimension, bias_region=[-bias_region, bias_region])
        if problem_name == 'ackley':
            prob_fct = func.DisAckley
        elif problem_name == 'sphere':
            prob_fct = func.DisSphere
        elif problem_name == 'rosenbrock':
            prob_fct = func.DisRosenbrock
        else:
            print('Wrong Function!')
            exit()

        # bias log
        bias_log.append(str(prob_i + start_index) + ',' + list2string(func.getBias()))
        print('function: ', problem_name, ', this bias: ', func.getBias())
        running_log.append('function: ' + problem_name + ', this bias: ' + list2string(func.getBias()))

        # optimization setting
        optimizer = RacosOptimization(dimension)

        positive_set = []
        negative_set = []
        new_sample_set = []
        label_set = []

        for repeat_i in range(repeat_num):
            print('repeat ', repeat_i, ' ----------------------------------------')
            running_log.append('repeat ' + str(repeat_i) + ' ----------------------------------------')

            # optimization process
            start_t = time.time()
            optimizer.mix_opt(obj_fct=prob_fct, ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability,
                              ub=uncertain_bits)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)

            # optimization results
            optimal = optimizer.get_optimal()
            print('optimal v: ', optimal.get_fitness(), ' - ', optimal.get_features())
            running_log.append('optimal v: ' + str(optimal.get_fitness()) + ' - ' + list2string(optimal.get_features()))
            print('spent time: ', hour, ':', minute, ':', second)
            running_log.append('spent time: ' + str(hour) + ':' + str(minute) + ':' + str(second))

            # log samples
            this_positive, this_negative, this_new, this_label = optimizer.get_log()

            print('sample number: ', len(this_positive), ':', len(this_label))
            running_log.append('sample number: ' + str(len(this_positive)) + ':' + str(len(this_label)))

            positive_set.extend(this_positive)
            negative_set.extend(this_negative)
            new_sample_set.extend(this_new)
            label_set.extend(this_label)
        print('----------------------------------------------')
        print('sample finish!')
        print('all sample number: ', len(positive_set), '-', len(negative_set), '-', len(new_sample_set), \
              '-', len(label_set))
        running_log.append('----------------------------------------------')
        running_log.append('all sample number: ' + str(len(positive_set)) + '-' + str(len(negative_set)) + '-'
                           + str(len(new_sample_set)) + '-' + str(len(label_set)))

        data_log_file = exp_path + str(problem_name) + '/dimension' + str(dimension_size) + '/DataLog/' + \
                        'data-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias' \
                        + str(bias_region) + '-' + str(start_index + prob_i) + '.pkl'
        bias_log_file = exp_path + str(problem_name) + '/dimension' + str(dimension_size) + '/RecordLog/' + 'bias-' \
                        + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias' + str(bias_region) \
                        + '-' + str(start_index + prob_i) + '.txt'
        running_log_file = exp_path + str(problem_name) + '/dimension' + str(dimension_size) + '/RecordLog/' + \
                           'running-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias' \
                           + str(bias_region) + '-' + str(start_index + prob_i) + '.txt'

        print('data logging: ', data_log_file)
        running_log.append('data log path: ' + data_log_file)
        save_log(positive_set, negative_set, new_sample_set, label_set, data_log_file)

        print('bias logging: ', bias_log_file)
        running_log.append('bias log path: ' + bias_log_file)
        fo.FileWriter(bias_log_file, bias_log, style='w')

        print('running logging: ', running_log_file)
        fo.FileWriter(running_log_file, running_log, style='w')


    return


def learning_data_construct():
    total_path = path + '/ExpLog/SyntheticProbsLog/'
    is_balance = True


    for prob_i in range(problem_num):

        print(problem_name, ' ', prob_i, ' ========================================')

        source_data_file = total_path + problem_name + '/dimension' + str(dimension_size) + '/DataLog/' + 'data-'\
                           + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias' + str(bias_region)\
                           + '-' + str(start_index + prob_i) + '.pkl'

        print('source data reading: ', source_data_file)
        pos_set, neg_set, new_set, label_set = read_log(source_data_file)

        print('constructing learning data')
        instance_set = learning_instance_construct(pos_set=pos_set, neg_set=neg_set, new_set=new_set)
        print('original data size: ', len(instance_set), ' - ', len(label_set))
        original_data = [instance_set, label_set]

        if is_balance is True:
            print('balancing data...')
            balance_instance_set, balance_label_set = learning_instance_balance(tensors=copy.deepcopy(instance_set),
                                                                                labels=copy.deepcopy(label_set))
            print('balanced data size: ', len(balance_instance_set), ' - ', len(balance_label_set))
            balance_data = [balance_instance_set, balance_label_set]
        else:
            print('skipping balance data')
            balance_data = None

        problem_log_file = total_path + str(problem_name) + '/dimension' + str(dimension_size) + '/RecordLog/'\
                           + 'bias-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias'\
                           + str(bias_region) + '-' + str(start_index + prob_i) + '.txt'

        strings = fo.FileReader(problem_log_file)
        this_string = strings[0].split(',')
        problem_index = int(this_string[0])
        problem_bias = string2list(this_string[1])

        # if problem_index != start_index + prob_i:
        #     print('problem index error!')
        #     exit(0)

        # learning data log
        # data format:
        # obj1: problem_name,problem_index: string
        # obj2: bias: list
        # obj3: original data: (instance, label)
        # obj4: balanced data: (instance, label) or None
        learning_data_file = total_path + str(problem_name) + '/dimension' + str(dimension_size) + '/LearningData/'\
                             + 'learning-data-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias'\
                             + str(bias_region) + '-' + str(start_index + prob_i) + '.pkl'
        print('learning data logging...')
        f = open(learning_data_file, 'wb')
        pickle.dump(problem_name + ',' + str(start_index + prob_i), f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(problem_bias, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(original_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(balance_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    return

def learning_exp():
    log_buffer = []

    random.seed(1)

    # training parameters
    epoch_size = 50
    batch_size = 32
    vali_rate = 0.1
    learn_rate = 0.0001
    categorical_size = 1
    validation_switch = True

    learner_path = path + '/ExpLearner/SyntheticProbsLearner/'
    data_path = path + '/ExpLog/SyntheticProbsLog/'
    print(problem_num)

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
        log_buffer.append('dimension size: ' + str(dimension_size))
        log_buffer.append('problem name: ' + problem_name)
        log_buffer.append('problem index: ' + str(start_index + prob_i))
        log_buffer.append('+++++++++++++++++++++++++++++++')

        log_name = learner_path + problem_name + '/dimension' + str(dimension_size) + '/TrainingLog/' + 'learning-log-' \
                   + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias' + str(bias_region) \
                   + '-' + str(start_index + prob_i) + '.txt'

        data_file = data_path + problem_name + '/dimension' + str(dimension_size) + '/LearningData/' + 'learning-data-' \
                    + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias' + str(bias_region) + '-' \
                    + str(start_index + prob_i) + '.pkl'

        print('data loading: ', data_file)
        data_inf, bias, ori_data, blc_data = learning_data_load(file_path=data_file)

        train_data, train_label = blc_data
        test_data, test_label = ori_data

        print('data transfer...')
        train_data = learning_data_transfer(instance_set=train_data)
        test_data = learning_data_transfer(instance_set=test_data)

        print('test data formulation: ', len(test_data), '*', len(test_data[0]), '*', len(test_data[0][0]), '*', \
              len(test_data[0][0][0]))
        log_buffer.append('--' + 'test data formulation: ' + str(test_data) + '*' + str(test_data[0]) + '*'
                          + str(test_data[0][0]) + '*' + str(test_data[0][0][0]) + '--')
        print('test label size: ', len(test_label))
        log_buffer.append('--' + 'test label size: ' + str(len(test_label)))
        print('train data formulation: ', len(train_data), '*', len(train_data[0]), '*', len(train_data[0][0]), '*', \
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

        net = ImageNet(middle_input_size=dimension_size, output_size=categorical_size)
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

                labels = labels.float()  # BCELoss used

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999 or i == len(trainloader) - 1:  # print every 2000 mini-batches
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

        net_file = learner_path + problem_name + '/dimension' + str(dimension_size) + '/DirectionalModel/' + 'learner-' \
                   + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias' + str(bias_region) \
                   + '-' + str(start_index + prob_i) + '.pkl'
        print('net saving...')
        torch.save(net, net_file)
        log_buffer.append('--net save: ' + net_file + '--')
        print('net saved!')

        fo.FileWriter(log_name, log_buffer, style='w')

    return


def run_exp_racos_for_synthetic_problem_analysis():

    # parameters
    positive_num = 2            # the set size of PosPop
    rand_probability = 0.99     # the probability of sample in model
    uncertain_bit = 1           # the dimension size that is sampled randomly
    adv_threshold = 10          # advance sample size

    opt_repeat = 10
    log_buffer = []

    # logging
    learner_path = path+'/ExpLearner/SyntheticProbsLearner/' + problem_name + '/dimension' + str(dimension_size)\
                   + '/DirectionalModel/' + 'learner-' + problem_name + '-' + 'dim' + str(dimension_size) + '-'\
                   + 'bias' + str(bias_region) + '-'
    problem_path = path+'/ExpLog/SyntheticProbsLog/' + problem_name + '/dimension' + str(dimension_size)\
                   + '/RecordLog/' + 'bias-' + problem_name + '-' + 'dim' + str(dimension_size) + '-'\
                   + 'bias' + str(bias_region) + '-'

    func = DistributedFunction(dimension, bias_region=[-bias_region, bias_region])
    target_bias = [0.1 for _ in range(dimension_size)]
    func.setBias(target_bias)

    if problem_name == 'ackley':
        prob_fct = func.DisAckley
    else:
        prob_fct = func.DisSphere

    relate_error_list = []
    net_ensemble = []

    for prob_i in range(problem_num):

        print(start_index + prob_i, '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        log_buffer.append(str(start_index + prob_i) + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        log_buffer.append('+++++++++++++++++++++++++++++++')
        log_buffer.append('optimization parameters')
        log_buffer.append('sample size: ' + str(sample_size))
        log_buffer.append('budget: ' + str(budget))
        log_buffer.append('positive num: ' + str(positive_num))
        log_buffer.append('random probability: ' + str(rand_probability))
        log_buffer.append('uncertain bits: ' + str(uncertain_bit))
        log_buffer.append('advance num: ' + str(adv_threshold))
        log_buffer.append('+++++++++++++++++++++++++++++++')
        log_buffer.append('problem parameters')
        log_buffer.append('dimension size: ' + str(dimension_size))
        log_buffer.append('problem name: ' + problem_name)
        log_buffer.append('bias_region: ' + str(bias_region))
        log_buffer.append('+++++++++++++++++++++++++++++++')

        problem_file = problem_path + str(start_index + prob_i) + '.txt'
        problem_str = fo.FileReader(problem_file)[0].split(',')
        problem_index = int(problem_str[0])
        problem_bias = string2list(problem_str[1])
        if problem_index != (start_index + prob_i):
            print('problem error!')
            exit(0)
        print('source bias: ', problem_bias)
        log_buffer.append('source bias: ' + list2string(problem_bias))

        reduisal = np.array(target_bias) - np.array(problem_bias)
        this_distance = reduisal * reduisal.T

        learner_file = learner_path + str(start_index + prob_i) + '.pkl'
        log_buffer.append('learner file: ' + learner_file)
        print('learner file: ', learner_file)

        net = torch.load(learner_file)

        net_list = [net]
        net_ensemble.append(net)

        opt_error_list = []

        for i in range(opt_repeat):

            print('optimize ', i, '===================================================')
            log_buffer.append('optimize ' + str(i) + '===================================================')

            exp_racos = ExpRacosOptimization(dimension, net_list)

            start_t = time.time()
            exp_racos.exp_mix_opt(obj_fct=prob_fct, ss=sample_size, bud=budget, pn=positive_num,
                                  rp=rand_probability, ub=uncertain_bit, at=adv_threshold)
            end_t = time.time()

            print('total budget is ', budget)
            log_buffer.append('total budget is ' + str(budget))

            hour, minute, second = time_formulate(start_t, end_t)
            print('spending time: ', hour, ':', minute, ':', second)
            log_buffer.append('spending time: ' + str(hour) + '+' + str(minute) + '+' + str(second))

            optimal = exp_racos.get_optimal()
            opt_error = optimal.get_fitness()
            optimal_x = optimal.get_features()

            opt_error_list.append(opt_error)
            print('validation optimal value: ', opt_error)
            log_buffer.append('validation optimal value: ' + str(opt_error))
            print('optimal x: ', optimal_x)
            log_buffer.append('optimal nn structure: ' + list2string(optimal_x))

        opt_mean = np.mean(np.array(opt_error_list))
        relate_error_list.append([this_distance, opt_mean])
        opt_std = np.std(np.array(opt_error_list))
        print('--------------------------------------------------')
        print('optimization result: ', opt_mean, '#', opt_std)
        log_buffer.append('--------------------------------------------------')
        log_buffer.append('optimization result: ' + str(opt_mean) + '#' + str(opt_std))


    result_path = path+'/Results/SyntheticProbs/' + problem_name + '/dimension' + str(dimension_size) + '/'
    relate_error_file = result_path + 'relate-error-' + problem_name + '-dim' + str(dimension_size) + '-bias' \
                        + str(bias_region) + '.txt'
    temp_buffer = []
    for i in range(len(relate_error_list)):
        relate, error = relate_error_list[i]
        temp_buffer.append(str(relate) + ',' + str(error))
    print('relate error logging: ', relate_error_file)
    log_buffer.append('relate error logging: ' + relate_error_file)
    fo.FileWriter(relate_error_file, temp_buffer, style='w')

    optimization_log_file = result_path + 'opt-log-' + problem_name + '-dim' + str(dimension_size) + '-bias' \
                            + str(bias_region) + '.txt'
    print('optimization logging: ', optimization_log_file)
    fo.FileWriter(optimization_log_file, log_buffer, style='w')



if __name__ == '__main__':
    # synthetic_problems_sample()
    # learning_data_construct()
    learning_exp()
    # run_exp_racos_for_synthetic_problem_analysis()
