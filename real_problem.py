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

    def __init__(self, prob_name='', prob_index=0, predictor=None):
        self.prob_name = prob_name
        self.prob_index = prob_index
        self.predictor = predictor
        return


def get_hyper_space():
    hyper_space = {
        'boosting_type': ('str', ('gbdt', 'rf', 'dart')),
        'learning_rate': ('float', (1e-6, 0.1)),
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
    region_array = []
    dimension.set_dimension_size(len(param_input))
    index = 0
    for k, (type, obj) in param_input.items():
        dimension.set_region(*label_coder.encode(type=type, index=index, key=k, objs=obj))
        index = index + 1
    return dimension, label_coder


# loading predictors
def get_predicotrs():
    dataset_30 = "australian,breast,electricity,buggyCrx,cmc,contraceptive,credit-a,GAMETES_Epistasis_2-Way_1000atts_0,GAMETES_Epistasis_2-Way_20atts_0,GAMETES_Epistasis_3-Way_20atts_0,GAMETES_Heterogeneity_20atts_1600_Het_0,Hill_Valley_without_noise,Hill_Valley_with_noise,mfeat-karhunen,mfeat-morphological,mfeat-pixel,mfeat-zernike,monk2,parity5+5,pima,tic-tac-toe,tokyo1,vehicle,wine-quality-red,yeast,airlines,titanic,twonorm,glass,horse-colic,messidor".split(
        ',')
    predictors = []
    nets = []
    print('Loading learner files...')

    for i, name in enumerate(dataset_30):
        learner_path = path + '/ExpLearner/RealProbsLearner/' + name + '/dimension11/DirectionalModel/'
        learner_file = learner_path + os.listdir(learner_path)[0]

        this_learner = torch.load(learner_file)
        nets.append(this_learner)
        this_predictor = ExpContainer(prob_name=name, prob_index=start_index + i,
                                      predictor=this_learner)
        predictors.append(this_predictor)

    print('Learner files loaded!')

    return predictors, nets


def run_for_synthetic_problem(problem_name):
    dtrain, dtest, dvalid = mlbp.get_train_test_data(problem_name)
    opt_error_list = []
    gen_error_list = []

    print('optimize ', problem_name, '===================================================')
    log_buffer.append('optimize ' + problem_name + '===================================================')

    for j in range(opt_repeat):
        print('============================ ', j, '================================')
        # log_buffer.append('============================ ' + str(j) + '=================================')

        exp_racos = ExpAdaRacosOptimization(dimension, expert)
        exp_racos.set_parameters(ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability, ub=uncertain_bit,
                                 at=adv_threshold)
        exp_racos.clear()
        start_t = time.time()
        x = exp_racos.sample()
        for i in range(budget):

            hyper_param = (sample_codec.sample_decode(x))
            model = lgb.LGBMClassifier()
            model.set_params(**hyper_param)
            bst = model.fit(dtrain[:, :-1], dtrain[:, -1])
            pred = bst.predict(dvalid[:, :-1])
            fitness = -f1_score(dvalid[:, -1], pred, average='macro')
            exp_racos.update_model(x, fitness)
            x = exp_racos.sample()
            if i % 10 == 0 and i > sample_size + positive_num:
                opt = exp_racos.get_optimal()
                # print '======================================================'
                print('budget ', i, ':', opt.get_fitness())
                # log_buffer.append('budget ' + str(i) + ':' + str(opt.get_fitness()))

        end_t = time.time()

        print('total budget is ', budget)
        # log_buffer.append('total budget is ' + str(budget))

        hour, minute, second = time_formulate(start_t, end_t)
        print('spending time: ', hour, ':', minute, ':', second)
        # log_buffer.append('spending time: ' + str(hour) + '+' + str(minute) + '+' + str(second))

        optimal = exp_racos.get_optimal()
        opt_error = optimal.get_fitness()
        optimal_x = optimal.get_features()
        hyper_param = (sample_codec.sample_decode(optimal_x))
        model = lgb.LGBMClassifier()
        model.set_params(**hyper_param)
        train = np.concatenate((dtrain, dvalid), axis=0)
        bst = model.fit(train[:, :-1], train[:, -1])
        pred = bst.predict(dtest[:, :-1])
        gen_error = -f1_score(dtest[:, -1], pred, average='macro')

        opt_error_list.append(opt_error)
        gen_error_list.append(gen_error)
        print('validation optimal value: ', opt_error)
        # log_buffer.append('validation optimal value: ' + str(opt_error))
        print('optimal x: ', optimal_x)
        # log_buffer.append('optimal nn structure: ' + list2string(optimal_x))

    opt_mean = np.mean(np.array(opt_error_list))
    opt_std = np.std(np.array(opt_error_list))
    gen_mean = np.mean(np.array(gen_error_list))
    gen_std = np.std(np.array(gen_error_list))

    return opt_mean, opt_std, gen_mean, gen_std


def run_for_real_problem_ave(problem_name):
    dtrain, dtest, dvalid = mlbp.get_train_test_data(problem_name)
    opt_error_list = []
    gen_error_list = []

    for j in range(opt_repeat):
        print('============================ ', j, '================================')
        # log_buffer.append('============================ ' + str(j) + '=================================')

        exp_racos = ExpRacosOptimization(dimension, nets)
        exp_racos.clear()
        start_t = time.time()
        model = lgb.LGBMClassifier()

        def score_fun(x):
            ## here is the score function
            hyper_param = (sample_codec.sample_decode(x))
            model.set_params(**hyper_param)
            bst = model.fit(dtrain[:, :-1], dtrain[:, -1])
            pred = bst.predict(dvalid[:, :-1])
            fitness = -f1_score(dvalid[:, -1], pred, average='macro')
            return fitness

        exp_racos.exp_mix_opt(obj_fct=score_fun, ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability,
                              ub=uncertain_bit,
                              at=adv_threshold)

        end_t = time.time()

        print('total budget is ', budget)
        # log_buffer.append('total budget is ' + str(budget))

        hour, minute, second = time_formulate(start_t, end_t)
        print('spending time: ', hour, ':', minute, ':', second)
        # log_buffer.append('spending time: ' + str(hour) + '+' + str(minute) + '+' + str(second))

        optimal = exp_racos.get_optimal()
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
        print('validation optimal value: ', opt_error)
        # log_buffer.append('validation optimal value: ' + str(opt_error))
        print('optimal x: ', optimal_x)
        # log_buffer.append('optimal nn structure: ' + list2string(optimal_x))

    opt_mean = np.mean(np.array(opt_error_list))
    opt_std = np.std(np.array(opt_error_list))
    gen_mean = np.mean(np.array(gen_error_list))
    gen_std = np.std(np.array(gen_error_list))

    return opt_mean, opt_std, gen_mean, gen_std


def run_for_real_problem_racos(problem_name):
    dtrain, dtest, dvalid = mlbp.get_train_test_data(problem_name)
    opt_error_list = []
    gen_error_list = []

    for j in range(opt_repeat):
        print('============================ ', j, '================================')
        # log_buffer.append('============================ ' + str(j) + '=================================')

        racos = RacosOptimization(dimension)
        racos.clear()
        start_t = time.time()
        model = lgb.LGBMClassifier()

        def score_fun(x):
            ## here is the score function
            hyper_param = (sample_codec.sample_decode(x))
            model.set_params(**hyper_param)
            bst = model.fit(dtrain[:, :-1], dtrain[:, -1])
            pred = bst.predict(dvalid[:, :-1])
            fitness = -f1_score(dvalid[:, -1], pred, average='macro')
            return fitness

        racos.mix_opt(obj_fct=score_fun, ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability,
                      ub=uncertain_bit)

        end_t = time.time()

        print('total budget is ', budget)
        # log_buffer.append('total budget is ' + str(budget))

        hour, minute, second = time_formulate(start_t, end_t)
        print('spending time: ', hour, ':', minute, ':', second)
        # log_buffer.append('spending time: ' + str(hour) + '+' + str(minute) + '+' + str(second))

        optimal = racos.get_optimal()
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
        print('validation optimal value: ', opt_error)
        # log_buffer.append('validation optimal value: ' + str(opt_error))
        print('optimal x: ', optimal_x)
        # log_buffer.append('optimal nn structure: ' + list2string(optimal_x))

    opt_mean = np.mean(np.array(opt_error_list))
    opt_std = np.std(np.array(opt_error_list))
    gen_mean = np.mean(np.array(gen_error_list))
    gen_std = np.std(np.array(gen_error_list))

    return opt_mean, opt_std, gen_mean, gen_std


if __name__ == '__main__':
    sample_size = 10  # the instance number of sampling in an iteration
    budget = 20  # budget in online style
    positive_num = 2  # the set size of PosPop
    rand_probability = 0.99  # the probability of sample in model
    uncertain_bit = 2  # the dimension size that is sampled randomly
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

    test_10 = "australian,breast,electricity,buggyCrx,cmc,contraceptive,credit-a,GAMETES_Epistasis_2-Way_1000atts_0,GAMETES_Epistasis_2-Way_20atts_0,GAMETES_Epistasis_3-Way_20atts_0,GAMETES_Heterogeneity_20atts_1600_Het_0,Hill_Valley_without_noise,Hill_Valley_with_noise,mfeat-karhunen,mfeat-morphological,mfeat-pixel,mfeat-zernike,monk2,parity5+5,pima,tic-tac-toe,tokyo1,vehicle,wine-quality-red,yeast,airlines,titanic,twonorm,glass,horse-colic,messidor".split(
        ',')
    test_10 += "adult,balance-scale,cnae,credit-g,crx,cylinder,flare,solar-flare_2,german".split(',')

    mlbp = BenchmarkHelper()
    predictors, nets = get_predicotrs()
    expert = Experts(predictors=predictors, eta=eta)

    res = []
    for problem_name in test_10:
        print('optimize ', problem_name, '===================================================')
        log_buffer.append('optimize ' + problem_name + '===================================================')

        rac_opt_mean, rac_opt_std, rac_gen_mean, rac_gen_std = run_for_real_problem_racos(problem_name)
        ave_opt_mean, ave_opt_std, ave_gen_mean, ave_gen_std = run_for_real_problem_ave(problem_name)
        ada_opt_mean, ada_opt_std, ada_gen_mean, ada_gen_std = run_for_synthetic_problem(problem_name)
        res.append([ada_opt_mean, ave_opt_mean, rac_opt_mean, ada_gen_mean, ave_gen_mean, rac_gen_mean])
        print('--------------------------------------------------')
        print('rac optimization result: ', rac_opt_mean, '#', rac_opt_std)
        print('ave optimization result: ', ave_opt_mean, '#', ave_opt_std)
        print('ada optimization result: ', ada_opt_mean, '#', ada_opt_std)
        print('rac generalization result: ', rac_gen_mean, '#', rac_gen_std)
        print('ave generalization result: ', ave_gen_mean, '#', ave_gen_std)
        print('ada generalization result: ', ada_gen_mean, '#', ada_gen_std)
        log_buffer.append('--------------------------------------------------')
        log_buffer.append('rac optimization result: ' + str(rac_opt_mean) + '#' + str(rac_opt_std))
        log_buffer.append('ave optimization result: ' + str(ave_opt_mean) + '#' + str(ave_opt_std))
        log_buffer.append('ada optimization result: ' + str(ada_opt_mean) + '#' + str(ada_opt_std))
        log_buffer.append('rac generalization result: ' + str(rac_gen_mean) + '#' + str(rac_gen_std))
        log_buffer.append('ave generalization result: ' + str(ave_gen_mean) + '#' + str(ave_gen_std))
        log_buffer.append('ada generalization result: ' + str(ada_gen_mean) + '#' + str(ada_gen_std))

    result_path = '/home/amax/Desktop/ExpAdaptation/ExpAdaptation/Results/RealProbs/'

    optimization_log_file = result_path + 'log-3mix-40probs-unbit2-budget-30.txt'
    print('optimization logging: ', optimization_log_file)
    fo.FileWriter(optimization_log_file, log_buffer, style='w')
    pd.DataFrame(columns=['ada-opt', 'exp-opt', 'racos-opt', 'ada-gen', 'exp-gen', 'racos-gen'], index=test_10,
                 data=res).to_csv(result_path + 'result-budget' + str(budget) + '.csv')
