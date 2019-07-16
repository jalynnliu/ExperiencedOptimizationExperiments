import pickle
import numpy as np
import random
import copy


def save_exp(envirs, labels, file_name):

    f = open(file_name, 'wb')
    pickle.dump(envirs, f, 2)
    pickle.dump(labels, f, 2)
    f.close()

    return


# generate environment through original data
def environment_generator(models, neg_sets, new_inses):

    if len(models) != len(neg_sets):
        print('data error!')
        exit(0)

    if True:
        environments = []
        for i in range(len(models)):
            ins = np.array(models[i])
            environment = []
            for j in range(len(neg_sets[i])):
                environment.append((np.array(neg_sets[i][j]) - ins).tolist())
            environment.append(new_inses[i])
            environments.append([environment])

    return environments


def exp_data_construct(dataset_list):

    tensor_3ds = []
    labels = []

    path = 'ec_logging/'

    for dataset_name in dataset_list:
        print('---------------------------------------------')
        print('from ', dataset_name, ' get exp...')

        exp_logging_name = path + dataset_name + '_log.pkl'

        f = open(exp_logging_name, 'rb')

        model = pickle.load(f)
        neg_set = pickle.load(f)
        new_inses = pickle.load(f)
        label = pickle.load(f)

        print('model size: ', len(model))
        print('neg set size: ', len(neg_set))
        print('label size: ', len(label), ' positive sample size:', sum(np.array(label)))

        f.close()

        environment = environment_generator(model, neg_set, new_inses)
        print('environment size: ', len(environment))

        tensor_3ds.extend(environment)
        labels.extend(label)

    print('==================================================')
    print('environments size: ', len(tensor_3ds))
    print('labels size: ', len(labels))

    file_name = path + 'exp_new_ec_training_data.pkl'

    save_exp(tensor_3ds, labels, file_name)

    return


def transform_original_data_to_balance():

    ori_data_path = 'ec_logging/exp_new_ec_training_data.pkl'
    balance_data_path = 'ec_logging//exp_new_ec_balance_data.pkl'

    f = open(ori_data_path, 'rb')

    tensors = pickle.load(f)
    labels = pickle.load(f)

    f.close()

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
        index_m = random.randint(0, len(min_tensor)-1)
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

    save_exp(all_tensor, all_label, balance_data_path)
    return


if __name__ == '__main__':

    dataset_list = ['annealing', 'arcene', 'balanceScale', 'banknote', 'breast_cancer_wisconsin', 'car', 'chess', 'cmc',
                    'CNAE9', 'credit', 'cylinder', 'drug_consumption', 'ecoli', 'flag', 'german credit', 'glass',
                    'horse_colic', 'imageSegmentation_car', 'iris', 'madelon', 'messidor', 'seismic', 'wdbc', 'wpbc']

    print('constructing...')
    exp_data_construct(dataset_list)
    print('balancing...')
    transform_original_data_to_balance()

