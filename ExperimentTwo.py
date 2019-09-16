from ObjectiveFunction import DistributedFunction
from Components import Dimension
from Tools import list2string
import time
import FileOperator as fo
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import random
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from Run_Racos import time_formulate
from ExpDataProcess import learning_data_construct
from RunExpRacos import run_exp_racos_for_synthetic_problem_analysis
from Tools import string2list
from ExpRacos import ExpRacosOptimization
import os
from ExpLearn import ImageNet, learning_exp, learning_data_transfer, learning_data_load, split_data
from SyntheticProbsSample import synthetic_problems_sample

path = '/data/ExpAdaptation'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def learning_exp_ensemble():
    random.seed(1)

    # training parameters
    epoch_size = 50
    batch_size = 32
    vali_rate = 0.1
    learn_rate = 0.0005
    categorical_size = 1
    validation_switch = True

    # exp data parameters
    dim_size = 10
    problem_name = 'sphere'
    start_index = 2000
    bias_region = 0.5
    problem_num = 2000

    learner_path = path + '/ExpLearner/SyntheticProbsLearner/'
    data_path = path + '/ExpLog/SyntheticProbsLog/'

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
    log_buffer.append('problem index: ' + str(start_index + 1))
    log_buffer.append('+++++++++++++++++++++++++++++++')

    log_name = learner_path + problem_name + '/dimension' + str(dim_size) + '/TrainingLog/' + 'learning-log-' \
               + problem_name + '-' + 'dim' + str(dim_size) + '-' + 'bias' + str(bias_region) \
               + '-' + str(start_index + 1) + '.txt'
    train_data = []
    train_label = []

    for prob_i in range(problem_num):
        data_file = data_path + problem_name + '/dimension' + str(dim_size) + '/LearningData/' + 'learning-data-' \
                    + problem_name + '-' + 'dim' + str(dim_size) + '-' + 'bias' + str(bias_region) + '-' \
                    + str(prob_i) + '.pkl'

        print('data loading: ', data_file)
        data_inf, bias, ori_data, blc_data = learning_data_load(file_path=data_file)

        train_data_, train_label_ = blc_data
        # test_data_, test_label_ = ori_data

        train_data.append(train_data_)
        train_label.append(train_label_)

    print('data transfer...')
    train_data = learning_data_transfer(instance_set=train_data)

    print('train data formulation: ', len(train_data), '*', len(train_data[0]), '*', len(train_data[0][0]), '*',
          len(train_data[0][0][0]))
    log_buffer.append('--' + 'train data formulation: ' + str(len(train_data)) + '*' + str(len(train_data[0])) + '*'
                      + str(len(train_data[0][0])) + '*' + str(len(train_data[0][0][0])) + '--')
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

            labels = labels.float()  # BCELoss used

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
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


                        outputs = net(images)
                        outputs = outputs.cpu()

                        # for BCELoss
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
    net_end = time.time()
    hour, minute, second = time_formulate(net_start, net_end)
    print('train net time: ', hour, ':', minute, ':', second)
    log_buffer.append('train net time: ' + str(hour) + ':' + str(minute) + ':' + str(second))

    net_file = learner_path + problem_name + '/dimension' + str(dim_size) + '/DirectionalModel/' + 'learner-' \
               + problem_name + '-' + 'dim' + str(dim_size) + '-' + 'bias' + str(bias_region) \
               + '-' + str(start_index + 1) + 'alldata.pkl'
    print('net saving...')
    torch.save(net, net_file)
    log_buffer.append('--net save: ' + net_file + '--')
    print('net saved!')

    fo.FileWriter(log_name, log_buffer, style='w')

    return

def run_exp_racos_for_synthetic_problem_analysis_ensemble():
    # parameters
    sample_size = 10  # the instance number of sampling in an iteration
    budget = 50  # budget in online style
    positive_num = 2  # the set size of PosPop
    rand_probability = 0.99  # the probability of sample in model
    uncertain_bit = 1  # the dimension size that is sampled randomly
    adv_threshold = 10  # advance sample size

    opt_repeat = 10

    dimension_size = 10
    problem_name = 'ackley'
    problem_num = 2000
    start_index = 0
    bias_region = 0.5

    dimension = Dimension()
    dimension.set_dimension_size(dimension_size)
    dimension.set_regions([[-1.0, 1.0] for _ in range(dimension_size)], [0 for _ in range(dimension_size)])

    log_buffer = []

    # logging
    learner_path = path + '/ExpLearner/SyntheticProbsLearner/' + problem_name + '/dimension' + str(dimension_size) \
                   + '/DirectionalModel/' + 'learner-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' \
                   + 'bias' + str(bias_region) + '-'
    problem_path = path + '/ExpLog/SyntheticProbsLog/' + problem_name + '/dimension' + str(dimension_size) \
                   + '/RecordLog/' + 'bias-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' \
                   + 'bias' + str(bias_region) + '-'

    func = DistributedFunction(dimension, bias_region=[-0.5, 0.5])
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
        # if problem_index != (start_index + prob_i):
        #     print('problem error!')
        #     exit(0)
        print('source bias: ', problem_bias)
        log_buffer.append('source bias: ' + list2string(problem_bias))

        reduisal = np.array(target_bias) - np.array(problem_bias)
        this_distance = reduisal * reduisal.T

        learner_file = learner_path + str(start_index + prob_i) + '.pkl'
        log_buffer.append('learner file: ' + learner_file)
        print('learner file: ', learner_file)

        net = torch.load(learner_file)
        net_ensemble.append(net)

    opt_error_list = []

    for i in range(opt_repeat):
        print('optimize ', i, '===================================================')
        log_buffer.append('optimize ' + str(i) + '===================================================')
        exp_racos = ExpRacosOptimization(dimension, net_ensemble)

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

    result_path = path + '/Results/SyntheticProbs/ExperimentTwo/'
    relate_error_file = result_path + 'relate-error-' + problem_name + '-dim' + str(dimension_size) + '-bias' \
                        + str(bias_region) + '2000sphere-ensemble.txt'
    temp_buffer = []
    for i in range(len(relate_error_list)):
        relate, error = relate_error_list[i]
        temp_buffer.append(str(relate) + ',' + str(error))
    print('relate error logging: ', relate_error_file)
    log_buffer.append('relate error logging: ' + relate_error_file)
    fo.FileWriter(relate_error_file, temp_buffer, style='w')

    optimization_log_file = result_path + 'opt-log-' + problem_name + '-dim' + str(dimension_size) + '-bias' \
                            + str(bias_region) + '2000sphere-ensemble.txt'
    print('optimization logging: ', optimization_log_file)
    fo.FileWriter(optimization_log_file, log_buffer, style='w')

    log_buffer = []
    problem_name = 'ackley'

    # logging
    learner_path = path + '/ExpLearner/SyntheticProbsLearner/' + problem_name + '/dimension' + str(dimension_size) \
                   + '/DirectionalModel/' + 'learner-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' \
                   + 'bias' + str(bias_region) + '-'
    problem_path = path + '/ExpLog/SyntheticProbsLog/' + problem_name + '/dimension' + str(dimension_size) \
                   + '/RecordLog/' + 'bias-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' \
                   + 'bias' + str(bias_region) + '-'

    relate_error_list = []
    problem_num = 1000

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
        # if problem_index != (start_index + prob_i):
        #     print('problem error!')
        #     exit(0)
        print('source bias: ', problem_bias)
        log_buffer.append('source bias: ' + list2string(problem_bias))

        reduisal = np.array(target_bias) - np.array(problem_bias)
        this_distance = reduisal * reduisal.T

        learner_file = learner_path + str(start_index + prob_i) + '.pkl'
        log_buffer.append('learner file: ' + learner_file)
        print('learner file: ', learner_file)

        net = torch.load(learner_file)
        net_ensemble[1000 + prob_i] = net

    opt_error_list = []

    for i in range(opt_repeat):
        print('optimize ', i, '===================================================')
        log_buffer.append('optimize ' + str(i) + '===================================================')
        exp_racos = ExpRacosOptimization(dimension, net_ensemble)

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

    relate_error_file = result_path + 'relate-error-' + problem_name + '-dim' + str(dimension_size) + '-bias' \
                        + str(bias_region) + 'half-sphere-half-ackley-ensemble.txt'
    temp_buffer = []
    for i in range(len(relate_error_list)):
        relate, error = relate_error_list[i]
        temp_buffer.append(str(relate) + ',' + str(error))
    print('relate error logging: ', relate_error_file)
    log_buffer.append('relate error logging: ' + relate_error_file)
    fo.FileWriter(relate_error_file, temp_buffer, style='w')

    optimization_log_file = result_path + 'opt-log-' + problem_name + '-dim' + str(dimension_size) + '-bias' \
                            + str(bias_region) + 'half-sphere-half-ackley-ensemble.txt'
    print('optimization logging: ', optimization_log_file)
    fo.FileWriter(optimization_log_file, log_buffer, style='w')


def run_exp_racos_for_synthetic_problem_analysis_remix():
    # parameters
    sample_size = 10  # the instance number of sampling in an iteration
    budget = 50  # budget in online style
    positive_num = 2  # the set size of PosPop
    rand_probability = 0.99  # the probability of sample in model
    uncertain_bit = 1  # the dimension size that is sampled randomly
    adv_threshold = 10  # advance sample size

    opt_repeat = 10

    dimension_size = 100
    problem_name = 'ackley'
    problem_num = 2000
    start_index = 0
    bias_region = 0.5

    dimension = Dimension()
    dimension.set_dimension_size(dimension_size)
    dimension.set_regions([[-1.0, 1.0] for _ in range(dimension_size)], [0 for _ in range(dimension_size)])

    log_buffer = []

    # logging

    func = DistributedFunction(dimension, bias_region=[-0.5, 0.5])
    target_bias = [0.1 for _ in range(dimension_size)]
    func.setBias(target_bias)

    if problem_name == 'ackley':
        prob_fct = func.DisAckley
    else:
        prob_fct = func.DisSphere

    relate_error_list = []
    net_ensemble = []

    for prob_i in range(problem_num):
        if prob_i > 999:
            problem_name = 'sphere'
            start_index = -1000
        learner_path = path + '/ExpLearner/SyntheticProbsLearner/' + problem_name + '/dimension' + str(dimension_size) \
                       + '/DirectionalModel/' + 'learner-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' \
                       + 'bias' + str(bias_region) + '-'
        problem_path = path + '/ExpLog/SyntheticProbsLog/' + problem_name + '/dimension' + str(dimension_size) \
                       + '/RecordLog/' + 'bias-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' \
                       + 'bias' + str(bias_region) + '-'

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
        # if problem_index != (start_index + prob_i):
        #     print('problem error!')
        #     exit(0)
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

    problem_name = 'sphere'
    result_path = path + '/Results/SyntheticProbs/ExperimentTwo/'
    relate_error_file = result_path + 'relate-error-' + problem_name + '-dim' + str(dimension_size) + '-bias' \
                        + str(bias_region) + 'remix.txt'
    temp_buffer = []
    for i in range(len(relate_error_list)):
        relate, error = relate_error_list[i]
        temp_buffer.append(str(relate) + ',' + str(error))
    print('relate error logging: ', relate_error_file)
    log_buffer.append('relate error logging: ' + relate_error_file)
    fo.FileWriter(relate_error_file, temp_buffer, style='w')

    optimization_log_file = result_path + 'opt-log-' + problem_name + '-dim' + str(dimension_size) + '-bias' \
                            + str(bias_region) + 'remix.txt'
    print('optimization logging: ', optimization_log_file)
    fo.FileWriter(optimization_log_file, log_buffer, style='w')


if __name__ == '__main__':
    synthetic_problems_sample()
    learning_data_construct()
    learning_exp()
    learning_exp_ensemble()
    run_exp_racos_for_synthetic_problem_analysis()
    run_exp_racos_for_synthetic_problem_analysis_remix()
