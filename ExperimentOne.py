from ObjectiveFunction import DistributedFunction
from Components import Dimension
from Tools import list2string
import time
import FileOperator as fo
import torch
import numpy as np
import pickle
from Run_Racos import time_formulate
from ExpDataProcess import learning_data_construct
from ExpLearn import learning_exp, ImageNet
import copy
from Tools import string2list
from ExpRacos import ExpRacosOptimization
from SyntheticProbsSample import synthetic_problems_sample

path = '/data/ExpAdaptation'
sample_size = 10  # the instance number of sampling in an iteration
budget = 500  # budget in online style
positive_num = 2  # the set size of PosPop
rand_probability = 0.99  # the probability of sample in model
uncertain_bits = 2  # the dimension size that is sampled randomly

start_index = 0
problem_name = 'sphere'
problem_num = 2000 - start_index

repeat_num = 10

exp_path = path + '/ExpLog/SyntheticProbsLog/'

bias_region = 0.5

dimension_size = 10

dimension = Dimension()
dimension.set_dimension_size(dimension_size)
dimension.set_regions([[-1.0, 1.0] for _ in range(dimension_size)], [0 for _ in range(dimension_size)])


def run_exp_racos_for_synthetic_problem_analysis():

    # parameters
    positive_num = 2            # the set size of PosPop
    rand_probability = 0.99     # the probability of sample in model
    uncertain_bit = 1           # the dimension size that is sampled randomly
    adv_threshold = 10          # advance sample size

    opt_repeat = 10
    log_buffer = []
    budget = 50

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
    synthetic_problems_sample(budget=budget, problem_name=problem_name, problem_size=problem_num, max_bias=bias_region)
    learning_data_construct()
    learning_exp()
    run_exp_racos_for_synthetic_problem_analysis()
