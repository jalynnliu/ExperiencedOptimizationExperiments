from ExpAdaRacos import ExpAdaRacosOptimization, Experts
import numpy as np
from Components import Dimension
from ObjectiveFunction import DistributedFunction
import torch
from Tools import list2string, string2list
import FileOperator as fo
import time
from Run_Racos import time_formulate


class ExpContainer(object):

    def __init__(self, prob_name='', prob_index=0, predictor=None):
        self.prob_name = prob_name
        self.prob_index = prob_index
        self.predictor = predictor
        return


# loading predictors
def get_predicotrs():

    predictors = []
    log_buffer = []

    if True:

        problem_name = 'ackley'
        dimension_size = 10
        bias_region = 0.2
        learner_num = 200
        start_index = 0

        learner_path = './ExpLearner/SyntheticProbsLearner/' + problem_name + '/dimension' + str(dimension_size) \
                       + '/DirectionalModel/' + 'learner-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' \
                       + 'bias' + str(bias_region) + '-'

        log_buffer.append('problem name: ' + str(problem_name))
        log_buffer.append('dimension size: ' + str(dimension_size))
        log_buffer.append('bias region: ' + str(bias_region))
        log_buffer.append('learner start: ' + str(start_index))
        log_buffer.append('learner num: ' + str(learner_num))
        log_buffer.append('learner path: ' + learner_path)

        for learner_i in range(learner_num):

            learner_file = learner_path + str(start_index + learner_i) + '.pkl'

            this_learner = torch.load(learner_file)
            this_predictor = ExpContainer(prob_name=problem_name, prob_index=start_index + learner_i,
                                          predictor=this_learner)
            predictors.append(this_predictor)

    return predictors, log_buffer


def run_for_synthetic_problem():

    sample_size = 10            # the instance number of sampling in an iteration
    budget = 50                 # budget in online style
    positive_num = 2            # the set size of PosPop
    rand_probability = 0.99     # the probability of sample in model
    uncertain_bit = 1           # the dimension size that is sampled randomly
    adv_threshold = 10          # advance sample size

    opt_repeat = 3

    dimension_size = 10
    problem_name = 'ackley'
    bias_region = 0.2

    eta = 0.9

    dimension = Dimension()
    dimension.set_dimension_size(dimension_size)
    dimension.set_regions([[-1.0, 1.0] for _ in range(dimension_size)], [0 for _ in range(dimension_size)])

    log_buffer = []

    # problem define
    func = DistributedFunction(dimension, bias_region == [-0.2, 0.2])
    target_bias = [0.1 for _ in range(dimension_size)]
    func.setBias(target_bias)

    if problem_name == 'ackley':
        prob_fct = func.DisAckley
    else:
        prob_fct = func.DisSphere

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

    predictors, load_buffer = get_predicotrs()
    expert = Experts(predictors=predictors, eta=eta)
    log_buffer.extend(load_buffer)

    opt_error_list = []

    for i in range(opt_repeat):
        print('optimize ', i, '===================================================')
        log_buffer.append('optimize ' + str(i) + '===================================================')

        exp_racos = ExpAdaRacosOptimization(dimension, expert)

        start_t = time.time()
        exp_racos.exp_ada_mix_opt(obj_fct=prob_fct, ss=sample_size, bud=budget, pn=positive_num,
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
    opt_std = np.std(np.array(opt_error_list))
    print('--------------------------------------------------')
    print('optimization result: ', opt_mean, '#', opt_std)
    log_buffer.append('--------------------------------------------------')
    log_buffer.append('optimization result: ' + str(opt_mean) + '#' + str(opt_std))

    result_path = './Results/SyntheticProbs/' + problem_name + '/dimension' + str(dimension_size) + '/'

    optimization_log_file = result_path + 'opt-log-' + problem_name + '-dim' + str(dimension_size) + '-bias' \
                            + str(bias_region) + '.txt'
    print('optimization logging: ', optimization_log_file)
    fo.FileWriter(optimization_log_file, log_buffer, style='w')

    return


if __name__ == '__main__':
    run_for_synthetic_problem()