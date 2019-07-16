from Racos import RacosOptimization
from ObjectiveFunction import DistributedFunction
from Components import Dimension
from Tools import list2string
from Run_Racos import time_formulate
import time
import pickle
import FileOperator as fo


# save experience log
def save_log(pos_set, neg_set, new_set, label_set, file_name):
    f = open(file_name, 'wb')
    pickle.dump(pos_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(neg_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(new_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(label_set, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    return


# load experience log
def read_log(file_name):

    f = open(file_name, 'rb')
    pos_set = pickle.load(f)
    neg_set = pickle.load(f)
    new_set = pickle.load(f)
    label_set = pickle.load(f)
    f.close()

    return pos_set, neg_set, new_set, label_set


# experience sample
def synthetic_probems_sample():

    sample_size = 10             # the instance number of sampling in an iteration
    budget = 500                # budget in online style
    positive_num = 2             # the set size of PosPop
    rand_probability = 0.99      # the probability of sample in model
    uncertain_bits = 2           # the dimension size that is sampled randomly

    start_index = 0
    problem_size = 5
    problem_name = 'sphere'

    repeat_num = 10

    exp_path = './ExpLog/SyntheticProbsLog/'

    bias_region = [-0.5, 0.5]

    dimension_size = 10

    dimension = Dimension()
    dimension.set_dimension_size(dimension_size)
    dimension.set_regions([[-1.0, 1.0] for _ in range(dimension_size)], [0 for _ in range(dimension_size)])

    for prob_i in range(problem_size):

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
        running_log.append(problem_name + ': ' + str(start_index + prob_i) + ' ==============================================')

        # problem setting
        func = DistributedFunction(dim=dimension, bias_region=bias_region)
        if problem_name == 'ackley':
            prob = func.DisAckley
        else:
            prob = func.DisSphere

        # bias log
        bias_log.append(str(prob_i) + ',' + list2string(func.getBias()))
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
            optimizer.mix_opt(obj_fct=prob, ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability, ub=uncertain_bits)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)

            # optimization results
            optimal = optimizer.get_optimal()
            print('optimal v: ', optimal.get_fitness(), ' - ', optimal.get_features())
            running_log.append('optimal v: '+ str(optimal.get_fitness()) + ' - ' + list2string(optimal.get_features()))
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
        print('all sample number: ', len(positive_set), '-', len(negative_set), '-', len(new_sample_set),\
            '-', len(label_set))
        running_log.append('----------------------------------------------')
        running_log.append('all sample number: ' + str(len(positive_set)) + '-' + str(len(negative_set)) + '-'
                           + str(len(new_sample_set)) + '-' + str(len(label_set)))

        data_log_file = exp_path + str(problem_name) + '/dimension' + str(dimension_size) + '/DataLog/' +\
                        'data-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias'\
                        + str(bias_region[1]) + '-' +str(start_index + prob_i) + '.pkl'
        bias_log_file = exp_path + str(problem_name) + '/dimension' + str(dimension_size) + '/RecordLog/' + 'bias-'\
                        + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias' + str(bias_region[1])\
                        + '-' +str(start_index + prob_i) + '.txt'
        running_log_file = exp_path + str(problem_name) + '/dimension' + str(dimension_size) + '/RecordLog/' +\
                           'running-' + problem_name + '-' + 'dim' + str(dimension_size) + '-' + 'bias'\
                           + str(bias_region[1]) + '-' +str(start_index + prob_i) + '.txt'

        print('data logging: ', data_log_file)
        running_log.append('data log path: ' + data_log_file)
        save_log(positive_set, negative_set, new_sample_set, label_set, data_log_file)

        print('bias logging: ', bias_log_file)
        running_log.append('bias log path: ' + bias_log_file)
        fo.FileWriter(bias_log_file, bias_log, style='w')

        print('running logging: ', running_log_file)
        fo.FileWriter(running_log_file, running_log, style='w')

    return


if __name__ == '__main__':

    synthetic_probems_sample()