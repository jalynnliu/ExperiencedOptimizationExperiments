from ExpAdaRacos import ExpAdaRacosOptimization, Experts
import numpy as np
from Components import Dimension
from ObjectiveFunction import DistributedFunction
from Tools import list2string
import FileOperator as fo
import time
from Run_Racos import time_formulate
import torch
from Racos import RacosOptimization
from ExpRacos import ExpRacosOptimization
from ExpLearn import ImageNet
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

path = '/data/ExpAdaptation'
sample_size = 10  # the instance number of sampling in an iteration
budget = 50  # budget in online style
positive_num = 2  # the set size of PosPop
rand_probability = 0.99  # the probability of sample in model
uncertain_bit = 1  # the dimension size that is sampled randomly
adv_threshold = 10  # advance sample size
dimension_size = 10
opt_repeat = 10

problem_name = 'rosenbrock'
bias_region = 0.5
learner_name = 'sphere'
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


def get_mixed_predicotrs():
    predictors = []

    if True:
        learner_name = 'sphere'
        learner_num = 2000
        sort_q = []

        print('Loading learner files...')

        for learner_i in range(learner_num):
            learner_path = path + '/ExpLearner/SyntheticProbsLearner/' + learner_name + '/dimension' + str(
                dimension_size) \
                           + '/DirectionalModel/' + 'learner-' + learner_name + '-' + 'dim' + str(dimension_size) + '-' \
                           + 'bias' + str(bias_region) + '-'
            bias_path = path + '/ExpLog/SyntheticProbsLog/' + learner_name + '/dimension' + str(dimension_size) \
                        + '/RecordLog/' + 'bias-' + learner_name + '-' + 'dim' + str(dimension_size) + '-' \
                        + 'bias' + str(bias_region) + '-'

            learner_file = learner_path + str(start_index + learner_i) + '.pkl'
            bias_file = bias_path + str(start_index + learner_i) + '.txt'

            biases = open(bias_file).readline()
            biases = biases.split(',')[1].split(' ')
            dist = 0
            for bias in biases:
                dist += abs(float(bias) - 0.1)

            this_learner = torch.load(learner_file)
            this_predictor = ExpContainer(prob_name=learner_name, prob_index=start_index + learner_i,
                                          predictor=this_learner, dist=dist)

            sort_q.append((learner_i, dist, this_predictor))
        sort_q.sort(key=lambda a: a[1])
        predictors += [x[2] for x in sort_q]

        learner_name = 'rosenbrock'
        sort_q = []
        learner_num = 1000

        for learner_i in range(learner_num):
            learner_path = path + '/ExpLearner/SyntheticProbsLearner/' + learner_name + '/dimension' + str(
                dimension_size) \
                           + '/DirectionalModel/' + 'learner-' + learner_name + '-' + 'dim' + str(dimension_size) + '-' \
                           + 'bias' + str(bias_region) + '-'
            bias_path = path + '/ExpLog/SyntheticProbsLog/' + learner_name + '/dimension' + str(dimension_size) \
                        + '/RecordLog/' + 'bias-' + learner_name + '-' + 'dim' + str(dimension_size) + '-' \
                        + 'bias' + str(bias_region) + '-'

            learner_file = learner_path + str(start_index + learner_i) + '.pkl'
            bias_file = bias_path + str(start_index + learner_i) + '.txt'

            biases = open(bias_file).readline()
            biases = biases.split(',')[1].split(' ')
            dist = 0
            for bias in biases:
                dist += abs(float(bias) - 0.1)

            this_learner = torch.load(learner_file)
            this_predictor = ExpContainer(prob_name=learner_name, prob_index=start_index + learner_i,
                                          predictor=this_learner, dist=dist)
            predictors.append(this_predictor)

            sort_q.append((learner_i, dist, this_predictor))
        sort_q.sort(key=lambda a: a[1])
        predictors += [x[2] for x in sort_q]
        predictor = []

        for i in range(10):
            predictor += predictors[i * step + 2000:i * step + step + 2000]
            predictor += predictors[i * 2 * step:i * 2 * step + step]


        print('Learner files loaded!')

    return predictor, [x.predictor for x in predictor]


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


def run(type):
    opt_error_list = []
    log_buffer.append('+++++++++++++++++++++++++++++++')
    log_buffer.append('Running: ' + type)
    log_buffer.append('+++++++++++++++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++')
    print('Running: ' + type)
    print('+++++++++++++++++++++++++++++++')
    if type == 'ada':
        # pre=sorted(predictors,key=lambda a:a.dist)
        expert = Experts(predictors=predictors, eta=eta, bg=budget)

    for i in range(opt_repeat):
        print('optimize ', i, '===================================================')
        log_buffer.append('optimize ' + str(i) + '===================================================')
        start_t = time.time()
        if type == 'ave':
            exp_racos = ExpRacosOptimization(dimension, nets)
            opt_error = exp_racos.exp_mix_opt(obj_fct=prob_fct, ss=sample_size, bud=budget, pn=positive_num,
                                              rp=rand_probability, ub=uncertain_bit, at=adv_threshold)
        elif type == 'ada':
            exp_racos = ExpAdaRacosOptimization(dimension, expert)
            opt_error = exp_racos.exp_ada_mix_opt(obj_fct=prob_fct, ss=sample_size, bud=budget, pn=positive_num,
                                                  rp=rand_probability, ub=uncertain_bit, at=adv_threshold, step=step)
        elif type == 'ground truth':
            exp_racos = ExpRacosOptimization(dimension, nets[:step])
            exp_racos.exp_mix_opt(obj_fct=prob_fct, ss=sample_size, bud=budget, pn=positive_num,
                                  rp=rand_probability, ub=uncertain_bit, at=adv_threshold)
        else:
            print('Wrong type!')
            return

        end_t = time.time()

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

    opt_mean = np.mean(np.array(opt_error_list), axis=0)
    opt_std = np.std(np.array(opt_error_list), axis=0)
    print('--------------------------------------------------')
    print('optimization result for ' + str(opt_repeat) + ' times average: ', opt_mean, ', standard variance is: ',
          opt_std)
    log_buffer.append('--------------------------------------------------')
    log_buffer.append('optimization result for ' + str(opt_repeat) + ' times average: ' + str(
        opt_mean) + ', standard variance is: ' + str(opt_std))

    return opt_mean, opt_std


def run_no_expert():
    log_buffer.append('+++++++++++++++++++++++++++++++')
    log_buffer.append('Running: no experts, pure Racos')
    log_buffer.append('+++++++++++++++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++')
    print('Running: no experts, pure Racos')
    print('+++++++++++++++++++++++++++++++')

    # optimization
    racos = RacosOptimization(dimension)
    opt_error_list = []

    for i in range(opt_repeat):
        start_t = time.time()
        racos.mix_opt(prob_fct, ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability,
                      ub=uncertain_bit)
        end_t = time.time()

        optimal = racos.get_optimal()
        opt_error = optimal.get_fitness()

        hour, minute, second = time_formulate(start_t, end_t)

        print('spending time: ', hour, ' hours ', minute, ' minutes ', second, ' seconds')
        print('optimal value: ', opt_error)
        opt_error_list.append(opt_error)
        print('validation optimal value: ', opt_error)
        log_buffer.append('validation optimal value: ' + str(opt_error))

    opt_mean = np.mean(np.array(opt_error_list))
    opt_std = np.std(np.array(opt_error_list))
    print('--------------------------------------------------')
    print('optimization result for ' + str(opt_repeat) + ' times average: ', opt_mean, ', standard variance is: ',
          opt_std)
    log_buffer.append('--------------------------------------------------')
    log_buffer.append('optimization result for ' + str(opt_repeat) + ' times average: ' + str(
        opt_mean) + ', standard variance is: ' + str(opt_std))

    return opt_mean, opt_std


if __name__ == '__main__':
    dimension = Dimension()
    dimension.set_dimension_size(dimension_size)
    dimension.set_regions([[-0.5, 0.5] for _ in range(dimension_size)], [0 for _ in range(dimension_size)])

    # problem define
    func = DistributedFunction(dimension, bias_region=[-bias_region, bias_region])
    target_bias = [0.4 for _ in range(dimension_size)]
    func.setBias(target_bias)

    if 'ackley' in problem_name:
        prob_fct = func.DisAckley
    elif 'sphere' in problem_name:
        prob_fct = func.DisSphere
    elif 'rosenbrock' in problem_name:
        prob_fct = func.DisRosenbrock
    else:
        print('Wrong function!')
        exit()

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
    log_buffer.append('bias: ' + list2string(target_bias))
    log_buffer.append('+++++++++++++++++++++++++++++++')

    predictors, nets = get_mixed_predicotrs()

    opt_mean_gt, opt_std_gt = run('ground truth')
    opt_mean_ada, opt_std_ada = run('ada')
    opt_mean_ave, opt_std_ave = run('ave')
    opt_mean_ne, opt_std_ne = run_no_expert()
    x = [i for i in range(len(opt_mean_ada))]
    y0 = [opt_mean_ne for _ in range(len(opt_mean_ada))]
    plt.plot(x, y0)
    plt.plot(x, opt_mean_ada)
    plt.plot(x, opt_mean_ave)
    plt.show()

    print(
        'We got the final results for ' + problem_name + ' with ' + str(learner_num) + ' ' + learner_name + ' experts')
    log_buffer.append(
        'We got the final results for ' + problem_name + ' with ' + str(learner_num) + ' ' + learner_name + ' experts')
    print('optimization result for ground truth: ', opt_mean_gt, ', standard variance is: ', opt_std_gt)
    log_buffer.append(
        'optimization result for ground truth: ' + str(opt_mean_gt) + ', standard variance is: ' + str(opt_std_gt))
    print('optimization result for adaptive: ', opt_mean_ada, ', standard variance is: ', opt_std_ada)
    log_buffer.append(
        'optimization result for adaptive: ' + str(opt_mean_ada) + ', standard variance is: ' + str(opt_std_ada))
    print('optimization result for all predictors average: ', opt_mean_ave, ', standard variance is: ', opt_std_ave)
    log_buffer.append(
        'optimization result for all predictors average: ' + str(opt_mean_ave) + ', standard variance is: ' + str(
            opt_std_ave))
    print('optimization result for no experts pure Racos: ', opt_mean_ne, ', standard variance is: ', opt_std_ne)
    log_buffer.append(
        'optimization result for no experts pure Racos: ' + str(opt_mean_ne) + ', standard variance is: ' + str(
            opt_std_ne))

    result_path = path + '/Results/ExperimentThree/'

    optimization_log_file = result_path + 'opt-log-' + problem_name + '-with-' + str(
        learner_num) + learner_name + '-budget' + str(budget) + '-bias' + str(bias_region) + '.txt'
    print('optimization logging: ', optimization_log_file)
    fo.FileWriter(optimization_log_file, log_buffer, style='w')

