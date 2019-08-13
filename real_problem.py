from ExpAdaRacos import ExpAdaRacosOptimization, Experts
import numpy as np
from Components import Dimension
from Tools import list2string
import FileOperator as fo
import time
from Run_Racos import time_formulate
import torch
from Racos import RacosOptimization
from ExpRacos import ExpRacosOptimization
from ParamsHelper import ParamsHelper
import lightgbm as lgb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

path = '/data/ExpAdaptation'
sample_size = 10  # the instance number of sampling in an iteration
budget = 50  # budget in online style
positive_num = 2  # the set size of PosPop
rand_probability = 0.99  # the probability of sample in model
uncertain_bit = 1  # the dimension size that is sampled randomly
adv_threshold = 10  # advance sample size
opt_repeat = 10

problem_name = 'sphere_group-sample'
start_index = 0
learner_num = 2000
step = 100

eta = 0.9

log_buffer = []


class ExpContainer(object):

    def __init__(self, prob_name='', prob_index=0, predictor=None, dist=0):
        self.prob_name = prob_name
        self.prob_index = prob_index
        self.predictor = predictor
        self.dist = dist
        return


# loading predictors
def get_predicotrs():
    predictors = []
    sort_q = []
    nets = []

    if True:
        learner_path = path + '/ExpLearner/SyntheticProbsLearner/' + learner_name + '/dimension' + str(
            dimension_size) \
                       + '/DirectionalModel/' + 'learner-' + learner_name + '-' + 'dim' + str(dimension_size) + '-' \
                       + 'bias' + str(bias_region) + '-'
        bias_path = path + '/ExpLog/SyntheticProbsLog/' + learner_name + '/dimension' + str(dimension_size) \
                    + '/RecordLog/' + 'bias-' + learner_name + '-' + 'dim' + str(dimension_size) + '-' \
                    + 'bias' + str(bias_region) + '-'

        print('Loading learner files...')

        for learner_i in range(learner_num):

            learner_file = learner_path + str(start_index + learner_i) + '.pkl'
            bias_file = bias_path + str(start_index + learner_i) + '.txt'

            biases = open(bias_file).readline()
            biases = biases.split(',')[1].split(' ')
            dist = 0
            for bias in biases:
                dist += abs(float(bias) - 0.1)

            this_learner = torch.load(learner_file)
            nets.append(this_learner)
            this_predictor = ExpContainer(prob_name=learner_name, prob_index=start_index + learner_i,
                                          predictor=this_learner, dist=dist)
            predictors.append(this_predictor)

            sort_q.append((learner_i, dist))
        sort_q.sort(key=lambda a: a[1])
        index = [x[0] for x in sort_q]
        predictors = np.array(predictors)[index].tolist()
        nets = np.array(nets)[index].tolist()

        print('Learner files loaded!')

    return predictors, nets


def run_for_synthetic_problem():
    sample_size = 10  # the instance number of sampling in an iteration
    budget = 50  # budget in online style
    positive_num = 2  # the set size of PosPop
    rand_probability = 0.99  # the probability of sample in model
    uncertain_bit = 1  # the dimension size that is sampled randomly
    adv_threshold = 10  # advance sample size

    opt_repeat = 10

    eta = 0.9

    hyper_space = get_hyper_space()
    dimension, sample_codec = get_dimension(hyper_space)

    log_buffer = []

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
    log_buffer.append('problem name: ' + problem_name)
    log_buffer.append('+++++++++++++++++++++++++++++++')

    predictors, load_buffer = get_predicotrs()
    expert = Experts(predictors=predictors, eta=eta)
    log_buffer.extend(load_buffer)

    opt_error_list = []

    for i in range(opt_repeat):
        print('optimize ', i, '===================================================')
        log_buffer.append('optimize ' + str(i) + '===================================================')

        exp_racos = ExpAdaRacosOptimization(dimension, expert)
        start_t = time.time()
        x = exp_racos.sample()
        for i in range(budget):
            hyper_param = (sample_codec.sample_decode(x))

            bst = lgb.train(hyper_param, dtrain, numround)
            fitness = bst.predict(dtest)
            exp_racos.update_optimal(x, fitness)
            x = exp_racos.sample()

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
    opt_std = np.std(np.array(opt_error_list))
    print('--------------------------------------------------')
    print('optimization result: ', opt_mean, '#', opt_std)
    log_buffer.append('--------------------------------------------------')
    log_buffer.append('optimization result: ' + str(opt_mean) + '#' + str(opt_std))

    result_path = path + '/Results/Ada/' + problem_name + '/dimension' + str(dimension_size) + '/'

    optimization_log_file = result_path + 'opt-log-' + problem_name + '-dim' + str(dimension_size) + '-bias' \
                            + str(bias_region) + '.txt'
    print('optimization logging: ', optimization_log_file)
    fo.FileWriter(optimization_log_file, log_buffer, style='w')

    return


def get_hyper_space():
    hyper_space = {
        'boosting': ('str', ('gbdt', 'rf', 'dart', 'doss')),
        'num_thread': ('int', (1, 50)),
        'application': ('str', ('regression', 'binary', 'multi-class', 'cross-entropy', 'lambdarank')),
        'learning_rate': ('float', (1e-6, 0.1)),
        'num_leaves': ('int', (2, 1000)),
        'freature_fraction': ('float', (0, 1)),
        'bagging_fraction': ('float', (0, 1)),
        'bagging_freq': ('int', (1, 100)),
        'lambda_l1': ('float', (0, 1)),
        'lambda_l2': ('float', (0, 1))
    }
    return hyper_space


def get_dimension(param_input):
    '''
    get dimension params by param input
    :param param_input: params input
    :return: dimension and the label coder
    '''
    dimension = Dimension()
    label_coder = ParamsHelper()
    region_array = []
    dimension.set_dimension_size(len(param_input))
    index = 0
    for k, (type, obj) in param_input.items():
        dimension.set_region(*label_coder.encode(type=type, index=index, key=k, objs=obj))
        index = index + 1
    return dimension, label_coder


if __name__ == '__main__':
    result_path = path + '/Results/'

    optimization_log_file = result_path + 'opt-log-' + problem_name + '-with-' + str(
        learner_num) + learner_name + '-budget' + str(budget) + '-bias' + str(bias_region) + '.txt'
    print('optimization logging: ', optimization_log_file)
    fo.FileWriter(optimization_log_file, log_buffer, style='w')
