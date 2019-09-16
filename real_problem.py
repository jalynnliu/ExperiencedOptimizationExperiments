from ExpAdaRacos import ExpAdaRacosOptimization, Experts
import numpy as np
from Components import Dimension
import FileOperator as fo
import time
from Run_Racos import time_formulate
import torch
from Racos import RacosOptimization
from ExpRacos import ExpRacosOptimization
from ExpLearn import ImageNet
from ParamsHelper import ParamsHelper
import lightgbm as lgb
from sklearn.metrics import f1_score
from Tools import BenchmarkHelper
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

path = '/data/ExpAdaptation'
sample_size = 10  # the instance number of sampling in an iteration
budget = 30  # budget in online style
positive_num = 2  # the set size of PosPop
rand_probability = 0.99  # the probability of sample in model
uncertain_bit = 1  # the dimension size that is sampled randomly
adv_threshold = 10  # advance sample size
opt_repeat = 5
eta = 0.9
step = 1

index = 0

log_buffer = []
source_data = "australian,breast,electricity,buggyCrx,cmc,contraceptive,credit-a,GAMETES_Epistasis_2-Way_1000atts_0," \
              "GAMETES_Epistasis_2-Way_20atts_0,GAMETES_Epistasis_3-Way_20atts_0,GAMETES_Heterogeneity_20atts_1600_Het_0," \
              "Hill_Valley_without_noise,Hill_Valley_with_noise,mfeat-karhunen,mfeat-morphological,mfeat-pixel," \
              "mfeat-zernike,monk2,parity5+5,pima,tic-tac-toe,tokyo1,vehicle,wine-quality-red,yeast,airlines,titanic," \
              "twonorm,glass,horse-colic".split(',')
target_data = "messidor,adult,balance-scale,cnae,credit-g,crx,cylinder,flare,solar-flare_2,german".split(',')
method = ['racos', 'ave', 'ada']


class ExpContainer(object):

    def __init__(self, prob_name='', prob_index=0, predictor=None):
        self.prob_name = prob_name
        self.prob_index = prob_index
        self.predictor = predictor
        return


def get_hyper_space():
    hyper_space = {
        'boosting_type': ('str', ('gbdt', 'rf', 'dart')),
        'learning_rate': ('float', (0.01, 0.2)),
        'n_estimators': ('int', (50, 100)),
        'num_leaves': ('int', (2, 1000)),
        'colsample_bytree': ('float', (0.1, 1)),
        'subsample': ('float', (0.1, 1)),
        'subsample_freq': ('int', (1, 100)),
        'reg_alpha': ('float', (0.1, 1)),
        'reg_lambda': ('float', (0.1, 1)),
        'min_child_weight': ('float', (0, 0.01)),
        'min_child_samples': ('int', (10, 30)),
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
    dimension.set_dimension_size(len(param_input))
    index = 0
    for k, (type, obj) in param_input.items():
        dimension.set_region(*label_coder.encode(type=type, index=index, key=k, objs=obj))
        index = index + 1
    return dimension, label_coder


# loading predictors
def get_predicotrs():
    predictors = []
    nets = []
    print('Loading learner files...')

    for i, name in enumerate(source_data):
        learner_path = path + '/ExpLearner/RealProbsLearner/' + name + '/dimension11/DirectionalModel/'
        learner_file = learner_path + os.listdir(learner_path)[0]

        this_learner = torch.load(learner_file)
        nets.append(this_learner)
        this_predictor = ExpContainer(prob_name=name, prob_index=i, predictor=this_learner)
        predictors.append(this_predictor)

    print('Learner files loaded!')

    return predictors, nets


def run_for_real_problem(problem_name, type):
    dtrain, dtest, dvalid = mlbp.get_train_test_data(problem_name)
    opt_error_list = []
    gen_error_list = []
    print(type, ' optimize ', problem_name, '===================================================')
    log_buffer.append(type + ' optimize ' + problem_name + '===================================================')

    for j in range(opt_repeat):
        print(j)
        log_buffer.append(str(j))
        model = lgb.LGBMClassifier()
        start_t = time.time()

        def score_fun(x):
            ## here is the score function
            hyper_param = (sample_codec.sample_decode(x))
            model.set_params(**hyper_param)
            bst = model.fit(dtrain[:, :-1], dtrain[:, -1])
            pred = bst.predict(dvalid[:, :-1])
            fitness = -f1_score(dvalid[:, -1], pred, average='macro')
            return fitness

        if type == 'racos':
            optimizer = RacosOptimization(dimension)
            optimizer.clear()
            optimizer.mix_opt(obj_fct=score_fun, ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability,
                              ub=uncertain_bit)
        elif type == 'ave':
            optimizer = ExpRacosOptimization(dimension, nets)
            log = optimizer.exp_mix_opt(obj_fct=score_fun, ss=sample_size, bud=budget, pn=positive_num,
                                        rp=rand_probability,
                                        ub=uncertain_bit, at=adv_threshold)
            for line in log:
                log_buffer.append(line)
        elif type == 'ada':
            optimizer = ExpAdaRacosOptimization(dimension, expert)
            optimizer.clear()
            log = optimizer.exp_ada_mix_opt(obj_fct=score_fun, ss=sample_size, bud=budget, pn=positive_num,
                                            rp=rand_probability,
                                            ub=uncertain_bit, at=adv_threshold, step=step)
            for line in log:
                log_buffer.append(line)
        else:
            print('Wrong type!')
            return

        end_t = time.time()

        print('total budget is ', budget)
        log_buffer.append('total budget is ' + str(budget))

        hour, minute, second = time_formulate(start_t, end_t)
        print('spending time: ', hour, ':', minute, ':', second)
        log_buffer.append('spending time: ' + str(hour) + '+' + str(minute) + '+' + str(second))

        optimal = optimizer.get_optimal()
        opt_error = optimal.get_fitness()
        optimal_x = optimal.get_features()
        hyper_param = (sample_codec.sample_decode(optimal_x))
        model = lgb.LGBMClassifier()
        model.set_params(**hyper_param)
        train = np.concatenate((dtrain, dvalid), axis=0)
        bst = model.fit(train[:, :-1], train[:, -1])
        pred = bst.predict(dtest[:, :-1])
        gen_error = -f1_score(dtest[:, -1], pred, average='macro')

        gen_error_list.append(gen_error)
        opt_error_list.append(opt_error)
        print('***********validation optimal value: ', opt_error)
        log_buffer.append('***********validation optimal value: ' + str(opt_error))
        print('***********generalize optimal value: ', gen_error)
        log_buffer.append('***********generalize optimal value: ' + str(gen_error))
        print('optimal x: ', optimal_x)
        # log_buffer.append('optimal nn structure: ' + list2string(optimal_x))

    opt_mean = np.mean(np.array(opt_error_list))
    opt_std = np.std(np.array(opt_error_list))
    gen_mean = np.mean(np.array(gen_error_list))
    gen_std = np.std(np.array(gen_error_list))

    return -opt_mean, opt_std, -gen_mean, gen_std


if __name__ == '__main__':
    hyper_space = get_hyper_space()
    dimension, sample_codec = get_dimension(hyper_space)

    log_buffer.append('+++++++++++++++++++++++++++++++')
    log_buffer.append('optimization parameters')
    log_buffer.append('sample size: ' + str(sample_size))
    log_buffer.append('budget: ' + str(budget))
    log_buffer.append('positive num: ' + str(positive_num))
    log_buffer.append('random probability: ' + str(rand_probability))
    log_buffer.append('uncertain bits: ' + str(uncertain_bit))
    log_buffer.append('advance num: ' + str(adv_threshold))
    log_buffer.append('num of datasets in one group: ' + str(step))
    log_buffer.append('methods: ' + str(method))
    log_buffer.append('+++++++++++++++++++++++++++++++')

    mlbp = BenchmarkHelper()
    predictors, nets = get_predicotrs()
    expert = Experts(predictors=predictors, eta=eta, step=step)
    dataset = ['flare']  # source_data+target_data

    res = []
    col_name = []
    for problem_name in dataset:
        print('optimize ', problem_name, '===================================================')
        log_buffer.append('optimize ' + problem_name + '===================================================')
        line = []
        for type in method:
            opt_mean, opt_std, gen_mean, gen_std = run_for_real_problem(problem_name, type)
            print('--------------------------------------------------')
            print(type + ' optimization result: ', opt_mean, '#', opt_std)
            print(type + ' generalization result: ', gen_mean, '#', gen_std)
            log_buffer.append('--------------------------------------------------')
            log_buffer.append(type + ' optimization result: ' + str(opt_mean) + '#' + str(opt_std))
            log_buffer.append(type + ' generalization result: ' + str(gen_mean) + '#' + str(gen_std))
            col_name += [type + '-opt', type + '-opt-std', type + '-gen', type + '-gen-std']
            line += [opt_mean, opt_std, gen_mean, gen_std]
        res.append(line)

    result_path = path + '/Results/RealProbs/'

    optimization_log_file = result_path + 'running-log-method-' + str(len(method)) + '-budget-' + str(
        budget) + '-' + str(index) + '.txt'
    print('optimization logging: ', optimization_log_file)
    fo.FileWriter(optimization_log_file, log_buffer, style='w')
    pd.DataFrame(columns=col_name, index=dataset,
                 data=res).to_csv(
        result_path + 'result-method-' + str(len(method)) + '-budget-' + str(budget) + '-' + str(index) + '.csv')
