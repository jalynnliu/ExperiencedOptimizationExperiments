import numpy as np
from SyntheticProbsSample import read_log
import random
import copy
import FileOperator as fo
from Tools import string2list
import pickle

path = '/home/amax/Desktop/ExpAdaptation'

def learning_instance_construct(pos_set=None, neg_set=None, new_set=None):

    instance_num = len(pos_set)

    instance_set = []
    for i in range(instance_num):

        this_instance = []

        # contrelization
        this_pos = np.array(pos_set[i])
        this_neg_set = neg_set[i]
        for j in range(len(this_neg_set)):
            this_neg = np.array(this_neg_set[j])
            this_instance.append((this_neg - this_pos).tolist())

        # add new sample
        this_instance.append(new_set[i])

        instance_set.append(this_instance)

    return instance_set


def learning_instance_balance(tensors=None, labels=None):

    positive_tensors = []
    negative_tensors = []

    for i in range(len(tensors)):

        if labels[i] == 1:
            positive_tensors.append(tensors[i])
        else:
            negative_tensors.append(tensors[i])

    print('original positive tensor size: ', len(positive_tensors))
    print('original negative tensor size: ', len(negative_tensors))

    print('balancing...')

    if len(positive_tensors) < len(negative_tensors):
        maj_tensor = negative_tensors
        maj_label = 0
        min_tensor = positive_tensors
        min_label = 1

    else:
        maj_tensor = positive_tensors
        maj_label = 1
        min_tensor = negative_tensors
        min_label = 0

    less_size = len(maj_tensor) - len(min_tensor)

    add_min_tensor = []

    for i in range(less_size):
        index_m = random.randint(0, len(min_tensor) - 1)
        add_min_tensor.append(copy.deepcopy(min_tensor[index_m]))

    min_tensor.extend(add_min_tensor)

    print('mixing...')

    i = 0
    j = 0
    all_tensor = []
    all_label = []
    while i < len(maj_tensor) and j < len(min_tensor):
        choose = random.randint(0, 1)
        if choose == 0:
            all_tensor.append(maj_tensor[i])
            all_label.append(maj_label)
            i += 1
        else:
            all_tensor.append(min_tensor[j])
            all_label.append(min_label)
            j += 1
    while i < len(maj_tensor):
        all_tensor.append(maj_tensor[i])
        all_label.append(maj_label)
        i += 1
    while j < len(min_tensor):
        all_tensor.append(min_tensor[j])
        all_label.append(min_label)
        j += 1

    print('positive tensor size: ', sum(all_label))
    print('negative tensor size: ', len(all_tensor) - sum(all_label))

    return all_tensor, all_label


def learning_data_construct():
    total_path = path + '/ExpLog/SyntheticProbsLog/'

    problem_name = 'ackley'
    dimension_size = 10
    bias_region = 0.5
    start_index = 1000

    is_balance = True

    problem_num = 1000

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

        if problem_index != start_index + prob_i:
            print('problem index error!')
            exit(0)

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


# load learning data, data format:
# 1st: data information
# 2rd: bias
# 3nd: original data
# 4th: balanced data
def learning_data_load(file_path=''):

    f = open(file_path, 'rb')
    data_inf = pickle.load(f)
    bias = pickle.load(f)
    ori_data = pickle.load(f)
    balanced_data = pickle.load(f)
    f.close()

    return data_inf, bias, ori_data, balanced_data


if __name__ == '__main__':
    learning_data_construct()
    # file_path = path+'/ExpLog/SyntheticProbsLog/ackley/dimension10/LearningData/learning-data-ackley-dim10-bias0.2-0.pkl'
    # learning_data_load(file_path=file_path)












