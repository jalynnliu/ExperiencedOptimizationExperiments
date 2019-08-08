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
from multiprocessing import Pool

path = '/data/ExpAdaptation'
sample_size = 10  # the instance number of sampling in an iteration
budget = 20  # budget in online style
positive_num = 2  # the set size of PosPop
rand_probability = 0.99  # the probability of sample in model
uncertain_bit = 1  # the dimension size that is sampled randomly
adv_threshold = 10  # advance sample size
dimension_size = 10
opt_repeat = 10

problem_name = 'sphere'
bias_region = 0.5
learner_name = 'sphere'
start_index = 0
learner_num = 200

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


def run(type):
    dimension = Dimension()
    dimension.set_dimension_size(dimension_size)
    dimension.set_regions([[-1.0, 1.0] for _ in range(dimension_size)], [0 for _ in range(dimension_size)])

    # problem define
    func = DistributedFunction(dimension, bias_region=[-bias_region, bias_region])
    target_bias = [0.1 for _ in range(dimension_size)]
    func.setBias(target_bias)

    if problem_name == 'ackley':
        prob_fct = func.DisAckley
    elif problem_name == 'sphere':
        prob_fct = func.DisSphere
    elif problem_name == 'rosenbrock':
        prob_fct = func.DisRosenbrock
    else:
        print('Wrong function!')
        exit()
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
        if type == 'exp':
            exp_racos = ExpRacosOptimization(dimension, nets)
            opt_error = exp_racos.exp_mix_opt(obj_fct=prob_fct, ss=sample_size, bud=budget, pn=positive_num,
                                              rp=rand_probability, ub=uncertain_bit, at=adv_threshold)
        elif type == 'ada':
            exp_racos = ExpAdaRacosOptimization(dimension, expert)
            opt_error = exp_racos.exp_ada_mix_opt(obj_fct=prob_fct, ss=sample_size, bud=budget, pn=positive_num,
                                                  rp=rand_probability, ub=uncertain_bit, at=adv_threshold)
        else:
            print('Wrong type!')
            return

        end_t = time.time()

        hour, minute, second = time_formulate(start_t, end_t)
        print('spending time: ', hour, ':', minute, ':', second)
        log_buffer.append('spending time: ' + str(hour) + '+' + str(minute) + '+' + str(second))

        opt_error_list.append(opt_error)
        print('validation optimal value: ', opt_error)
        log_buffer.append('validation optimal value: ' + str(opt_error))

    opt_mean = np.mean(np.array(opt_error_list), axis=0)
    opt_std = np.std(np.array(opt_error_list), axis=0)
    print('--------------------------------------------------')
    print('optimization result for ' + str(opt_repeat) + ' times average: ', opt_mean, ', standard variance is: ',
          opt_std)
    log_buffer.append('--------------------------------------------------')
    log_buffer.append('optimization result for ' + str(opt_repeat) + ' times average: ' + str(
        opt_mean) + ', standard variance is: ' + str(opt_std))

    return opt_mean


def run_racos():
    # parameters
    sample_size = 10  # the instance number of sampling in an iteration
    budget = 500  # budget in online style
    positive_num = 2  # the set size of PosPop
    rand_probability = 0.99  # the probability of sample in model
    uncertain_bit = 1  # the dimension size that is sampled randomly
    bias_region = 0.5

    repeat = 10

    # dimension setting
    dimension_size = 10

    dimension = Dimension()
    dimension.set_dimension_size(dimension_size)
    dimension.set_regions([[-1.0, 1.0] for _ in range(dimension_size)], [0 for _ in range(dimension_size)])

    func = DistributedFunction(dim=dimension, bias_region=[-bias_region, bias_region])
    if problem_name == 'rosenbrock':
        prob = func.DisRosenbrock
    else:
        prob = func.DisSphere

    # optimization
    racos = RacosOptimization(dimension)
    opt_error_list = []

    for i in range(repeat):
        start_t = time.time()
        racos.mix_opt(prob, ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability, ub=uncertain_bit)
        end_t = time.time()

        optimal = racos.get_optimal()

        hour, minute, second = time_formulate(start_t, end_t)

        print('total budget is ', budget, '------------------------------')
        print('spending time: ', hour, ' hours ', minute, ' minutes ', second, ' seconds')
        print('optimal value: ', optimal.get_fitness())
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

    return opt_mean


if __name__ == "__main__":


    predictors, nets = get_predicotrs()

    y2 = run('ada')
    y1 = run('exp')
    pure_racos = run_racos()

    x = [i for i in range(len(y1))]
    y0 = [pure_racos for _ in range(len(y1))]
    plt.plot(x, y0)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()

    log_buffer.append('---------------------exp 200 budget result--------------------')
    log_buffer.append(str(y1))
    log_buffer.append('---------------------ada 200 budget result--------------------')
    log_buffer.append(str(y2))

    result_path = path + '/Results/'

    optimization_log_file = result_path + 'log-' + problem_name + '-dim' + str(dimension_size) + '-bias' \
                            + str(bias_region) + '-experiment4.txt'
    print('optimization logging: ', optimization_log_file)
    fo.FileWriter(optimization_log_file, log_buffer, style='w')
