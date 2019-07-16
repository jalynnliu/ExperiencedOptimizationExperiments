from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import FileOperator as fo
import numpy as np
import pickle
from Tools import list2string
import csv
from ObjectiveFunction import dataset_reader


def data_reader(file_name, feature_types):

    split_string = ','

    buffer = fo.FileReader(file_name)

    # # get feature types
    # split_x = buffer[0].split(split_string)
    # feature_size = len(split_x)
    # # true means number, False means string
    # feature_label = [True for i in xrange(feature_size)]
    # unlabeled_index = [i for i in xrange(feature_size)]
    # index = 0
    # while len(unlabeled_index) != 0:
    #     split_x = buffer[index].split(split_string)
    #     i = 0
    #     while i < len(unlabeled_index):
    #         if split_x[unlabeled_index[i]][0] != '?':
    #             if split_x[unlabeled_index[i]][0] < '0' or split_x[unlabeled_index[i]][0] > '9':
    #                 feature_label[unlabeled_index[i]] = False
    #             unlabeled_index.remove(unlabeled_index[i])
    #         else:
    #             i += 1
    #     index += 1
    #
    # print 'feature type-----------------------'
    # for i in xrange(len(feature_label)):
    #     if feature_label[i] is True:
    #         f_type = 'number'
    #     else:
    #         f_type = 'string'
    #     print i, ':', f_type
    # print '-----------------------------------'


    features = []
    # buffer = buffer[0].split('\r')     # special for HTUR2
    for i in range(len(buffer)):
        # if i % 1 == 0:
        #     print 'buffer ', i, ' processing...'
        split_x = buffer[i].split(split_string)
        feature = []
        for j in range(len(feature_types)):
            split_x[j] = split_x[j].replace('\n', '')
            split_x[j] = split_x[j].replace('\r', '')
            split_x[j] = split_x[j].replace(' ', '')
            if feature_types[j] is True and split_x[j] != '?':
                # print j, ',', i, ':', split_x[j]
                feature.append(float(split_x[j]))
            else:
                feature.append(split_x[j])
        features.append(feature)

    return features


def csv_data_reader(file_name, feature_types):
    csv_reader = csv.reader(open(file_name))
    features = []
    for row in csv_reader:
        feature = []
        for i in range(len(row)):
            if feature_types[i] is True and row[i] != '?':
                feature.append(float(row[i]))
            else:
                feature.append(row[i])
        features.append(feature)
    return features


def missing_value_processing(features, feature_types):

    print('missing value processing----------------------------')

    for i in range(len(feature_types)):
        # if the type is number, mean value will be used to replace missing value
        if feature_types[i] is True:
            values = []
            missing_index = []
            for j in range(len(features)):
                if features[j][i] == '?':
                    missing_index.append(j)
                else:
                    values.append(features[j][i])
            print('feature ', i, ' missing value number:', len(missing_index))
            mean_v = np.mean(np.array(values))
            for j in missing_index:
                features[j][i] = mean_v
        # if the type is categorical, the mode will be used to replace missing value
        else:
            values = []
            value_freq = []
            missing_index = []
            for j in range(len(features)):
                if features[j][i] == '?':
                    missing_index.append(j)
                else:
                    v_index = 0
                    while v_index < len(values):
                        if values[v_index] == features[j][i]:
                            value_freq[v_index] += 1
                            break
                        v_index += 1
                    if v_index == len(values):
                        values.append(features[j][i])
                        value_freq.append(1)
            print('feature ', i, ' missing value number:', len(missing_index))
            max_value = values[value_freq.index(max(value_freq))]
            for j in missing_index:
                features[j][i] = max_value

    return features


def categorical_feature_encoding(features, feature_types):
    print('categorical feature encoding...')
    for i in range(len(feature_types)):
        if feature_types[i] is False:
            feature = []
            for j in range(len(features)):
                feature.append(features[j][i])
            le = LabelEncoder()
            le = le.fit(feature)
            new_feature = le.transform(feature)
            for j in range(len(features)):
                features[j][i] = new_feature[j]
    print('feature encoding finished!')
    return features


def label_encoding(label):
    le = LabelEncoder()
    le.fit(label)
    new_label = le.transform(label).tolist()
    label_name = list(le.classes_)
    return new_label, label_name


def extracting_label(data, label_index):
    label = []
    for i in range(len(data)):
        label.append(data[i][label_index])
        del data[i][label_index]
    return data, label


def split_data(data, label, percent=0.2):
    k = int(1.0/percent)
    skf = StratifiedKFold(n_splits=k)
    for train_index, test_index in skf.split(data, label):
        train_data, test_data = np.array(data)[train_index].tolist(), np.array(data)[test_index].tolist()
        train_label, test_label = np.array(label)[train_index].tolist(), np.array(label)[test_index].tolist()
        break
    return train_data, train_label, test_data, test_label


# delete useless features from data
def feature_selected(data, del_index):
    new_data = []
    for f_i in range(len(data)):
        feature = []
        for i in range(len(data[f_i])):
            if not (i in del_index):
                feature.append(data[f_i][i])
        new_data.append(feature)
    return new_data


def dataset_info():

    dataset_list = ['abalone', 'adult', 'annealing', 'arcene', 'balanceScale', 'banknote', 'breast_cancer_wisconsin',
                    'car', 'chess', 'chess2', 'cmc', 'CNAE9', 'covtype', 'credit', 'cylinder', 'drug_consumption',
                    'ecoli', 'eeg', 'flag', 'german credit', 'gisette', 'glass', 'horse_colic', 'HTRU2',
                    'imageSegmentation_car', 'iris', 'jsbach', 'letterRecognition', 'madelon', 'magic04', 'messidor',
                    'mushroom', 'nursery', 'occupancy', 'seismic', 'spambase', 'statlogSegment', 'wdbc', 'wilt',
                    'wine_quality_red', 'wine_quality_white', 'wpbc', 'yeast', 'house_vote']

    feature_type_list = []
    useless_index_list = []
    label_index_list = []

    # abalone:
    feature_types = [False]
    for i in range(7):
        feature_types.append(True)
    feature_types.append(False)
    label_index = 8

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # adult
    feature_types = [True, False, True, False, True, False, False, False, False, False, True, True, True,
                     False, False]
    label_index = 14

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # annealing
    feature_types = [False for i in range(39)]
    number_index = [3, 4, 7, 12, 32, 33, 34, 37]
    for index in number_index:
        feature_types[index] = True
    useless_index = [10, 12, 13, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 37]
    label_index = 38

    feature_type_list.append(feature_types)
    useless_index_list.append(useless_index)
    label_index_list.append(label_index)

    # arcene
    feature_types = [True for i in range(10000)]
    label_index = len(feature_types)

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # balanceScale
    feature_types = [False, True, True, True, True]
    label_index = 0

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # banknote
    feature_types = [True, True, True, True, False]
    label_index = 4

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # breast_cancer_wisconsin
    feature_types = [False for i in range(11)]
    useless_index = [0]
    label_index = 10

    feature_type_list.append(feature_types)
    useless_index_list.append(useless_index)
    label_index_list.append(label_index)

    # car
    feature_types = [False, False, False, False, False, False, False]
    label_index = 6

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # chess
    feature_types = [False for i in range(37)]
    label_index = 36

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # chess2
    feature_types = [False for i in range(7)]
    label_index = 6

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # cmc
    feature_types = [True, False, False, True, False, False, False, False, False, False]
    label_index = 9

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # CNAE9
    feature_types = [False]
    for i in range(856):
        feature_types.append(True)
    label_index = 0

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # covtype
    feature_types = [True for i in range(10)]
    for i in range(45):
        feature_types.append(False)
    label_index = 54

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # credit
    feature_types = [False, True, True, False, False, False, False, True, False, False, True, False, False, True,
                     True, False]
    label_index = 15

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # cylinder
    feature_types = [True]
    for i in range(19):
        feature_types.append(False)
    for i in range(19):
        feature_types.append(True)
    feature_types.append(False)
    label_index = 39

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # drug_consumption
    feature_types = [False]
    for i in range(12):
        feature_types.append(True)
    for i in range(19):
        feature_types.append(False)
    useless_index = [0]
    label_index = 31

    feature_type_list.append(feature_types)
    useless_index_list.append(useless_index)
    label_index_list.append(label_index)

    # ecoli
    feature_types = [False, True, True, True, True, True, True, True, False]
    label_index = 8

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # eeg
    feature_types = [True for i in range(14)]
    feature_types.append(False)
    label_index = 14

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # flag
    feature_types = [False for i in range(30)]
    feature_types[3] = True
    feature_types[4] = True
    for i in range(3):
        feature_types[7 + i] = True
    for i in range(5):
        feature_types[18 + i] = True
    useless_index = [0]
    label_index = 6

    feature_type_list.append(feature_types)
    useless_index_list.append(useless_index)
    label_index_list.append(label_index)

    # german credit
    feature_types = [False for i in range(21)]
    feature_types[1] = True
    feature_types[4] = True
    feature_types[7] = True
    feature_types[10] = True
    feature_types[12] = True
    feature_types[15] = True
    feature_types[17] = True
    label_index = 20

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # gisette
    feature_types = [True for i in range(5000)]
    label_index = len(feature_types)

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # glass
    feature_types = [False]
    for i in range(9):
        feature_types.append(True)
    feature_types.append(False)
    useless_index = [0]
    label_index = 10

    feature_type_list.append(feature_types)
    useless_index_list.append(useless_index)
    label_index_list.append(label_index)

    # horse_colic
    feature_types = [False for i in range(28)]
    feature_types[3] = True
    feature_types[4] = True
    feature_types[5] = True
    feature_types[15] = True
    feature_types[18] = True
    feature_types[19] = True
    feature_types[21] = True
    useless_index = [2]
    label_index = 23

    feature_type_list.append(feature_types)
    useless_index_list.append(useless_index)
    label_index_list.append(label_index)

    # HTRU2
    feature_types = [True for i in range(8)]
    feature_types.append(False)
    label_index = 8

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # imageSegmentation_car
    feature_types = [True for i in range(20)]
    feature_types[0] = False
    label_index = 0

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # iris
    feature_types = [True, True, True, True, False]
    label_index = 4

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # jsbach
    feature_types = [False for i in range(17)]
    feature_types[15] = True
    label_index = 16

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # letterRecognition
    feature_types = [True for i in range(17)]
    feature_types[0] = False
    label_index = 0

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # madelon
    feature_types = [True for i in range(500)]
    label_index = len(feature_types)

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # magic04
    feature_types = [True for i in range(11)]
    feature_types[10] = False
    label_index = 10

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # messidor
    feature_types = [True for i in range(20)]
    feature_types[19] = False
    label_index = 19

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # mushroom
    feature_types = [False for i in range(23)]
    label_index = 0

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # nursery
    feature_types = [False for i in range(9)]
    label_index = 8

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # occupancy
    feature_types = [False, False, True, True, True, True, True, False]
    useless_index = [0, 1]
    label_index = 7

    feature_type_list.append(feature_types)
    useless_index_list.append(useless_index)
    label_index_list.append(label_index)

    # seismic
    feature_types = [True for i in range(19)]
    feature_types[0] = False
    feature_types[1] = False
    feature_types[2] = False
    feature_types[7] = False
    feature_types[18] = False
    label_index = 18

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # spambase
    feature_types = [True for i in range(58)]
    feature_types[57] = False
    label_index = 57

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # statlogSegment
    feature_types = [True for i in range(20)]
    feature_types[19] = False
    label_index = 19

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # wdbc
    feature_types = [True for i in range(32)]
    feature_types[0] = False
    feature_types[1] = False
    useless_index = [0]
    label_index = 1

    feature_type_list.append(feature_types)
    useless_index_list.append(useless_index)
    label_index_list.append(label_index)

    # wilt
    feature_types = [False, True, True, True, True, True]
    label_index = 0

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # wine_quality_red
    feature_types = [True for i in range(12)]
    feature_types[11] = False
    label_index = 11

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # wine_quality_white
    feature_types = [True for i in range(12)]
    feature_types[11] = False
    label_index = 11

    feature_type_list.append(feature_types)
    useless_index_list.append([])
    label_index_list.append(label_index)

    # wpbc
    feature_types = [True for i in range(35)]
    feature_types[0] = False
    feature_types[1] = False
    useless_index = [0]
    label_index = 1

    feature_type_list.append(feature_types)
    useless_index_list.append(useless_index)
    label_index_list.append(label_index)

    # yeast
    feature_types = [False, True, True, True, True, True, True, True, True, False]
    useless_index = [0]
    label_index = 9

    feature_type_list.append(feature_types)
    useless_index_list.append(useless_index)
    label_index_list.append(label_index)

    # house_vote
    feature_types = [False for i in range(17)]
    useless_index = []
    label_index = 0

    feature_type_list.append(feature_types)
    useless_index_list.append(useless_index)
    label_index_list.append(label_index)

    path = 'data_set/'

    for i in range(len(dataset_list)):

        print('-----------------------------------------------------------------------------')
        print('data set:', dataset_list[i])

        train_data_file = path + dataset_list[i] + '/' + dataset_list[i] + '_train_data.pkl'
        test_data_file = path + dataset_list[i] + '/' + dataset_list[i] + '_test_data.pkl'

        train_feature, train_label, test_feature, test_label = dataset_reader(train_data_file, test_data_file)

        label_index = label_index_list[i]
        feature_types = feature_type_list[i]
        useless_index = useless_index_list[i]

        if label_index < len(feature_types):
            del feature_types[label_index]

        new_feature_types = []
        for j in range(len(feature_types)):
            if not (j in useless_index):
                new_feature_types.append(feature_types[j])
        feature_types = new_feature_types

        true_c = 0
        for j in range(len(feature_types)):
            if feature_types[j] is True:
                true_c += 1
        false_c = len(feature_types) - true_c

        print('training data size: ', train_feature.shape[0])
        print('training label size: ', train_label.shape[0])
        print('testing data size: ', test_feature.shape[0])
        print('testing label size: ', test_label.shape[0])

        print('dimension size: ', train_feature.shape[1])

        print('numeircal diemsnion size: ', true_c)
        print('categorical dimension size: ', false_c)


def dataset_processing():
    path = 'data_set/'
    # feature type False means categorical and True means number
    # cylinder
    if False:
        dataset_name = 'cylinder'
        ori_file = 'bands.data.txt'
        feature_types = [True]
        for i in range(19):
            feature_types.append(False)
        for i in range(19):
            feature_types.append(True)
        feature_types.append(False)
        label_index = 39

    # abalone
    if False:
        dataset_name = 'abalone'
        ori_file = 'abalone.data.txt'
        feature_types = [False]
        for i in range(7):
            feature_types.append(True)
        feature_types.append(False)
        label_index = 8

    # anneal
    if False:
        dataset_name = 'annealing'
        ori_train_file = 'anneal.data.txt'
        ori_test_file = 'anneal.test.txt'
        feature_types = [False for i in range(39)]
        number_index = [3, 4, 7, 12, 32, 33, 34, 37]
        for index in number_index:
            feature_types[index] = True
        useless_index = [10, 12, 13, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 37]
        label_index = 38

    # balanceScale
    if False:
        dataset_name = 'balanceScale'
        ori_file = 'balance-scale.data.txt'
        feature_types = [False, True, True, True, True]
        label_index = 0

    # banknote
    if False:
        dataset_name = 'banknote'
        ori_file = 'data_banknote_authentication.txt'
        feature_types = [True, True, True, True, False]
        label_index = 4

    # car
    if False:
        dataset_name = 'car'
        ori_file = 'car.data.txt'
        feature_types = [False, False, False, False, False, False, False]
        label_index = 6

    # chess
    if False:
        dataset_name = 'chess'
        ori_file = 'chess.data.txt'
        feature_types = [False for i in range(37)]
        label_index = 36

    # chess2
    if False:
        dataset_name = 'chess2'
        ori_file = 'krkopt.data.txt'
        feature_types = [False for i in range(7)]
        label_index = 6

    # cmc
    if False:
        dataset_name = 'cmc'
        ori_file = 'cmc.data.txt'
        feature_types = [True, False, False, True, False, False, False, False, False, False]
        label_index = 9

    # CNAE9
    if False:
        dataset_name = 'CNAE9'
        ori_file = 'CNAE-9.data.txt'
        feature_types = [False]
        for i in range(856):
            feature_types.append(True)
        label_index = 0

    # credit
    if False:
        dataset_name = 'credit'
        ori_file = 'crx.data.txt'
        feature_types = [False, True, True, False, False, False, False, True, False, False, True, False, False, True,
                         True, False]
        label_index = 15

    # egg
    if False:
        dataset_name = 'eeg'
        ori_file = 'EEG_Eye_State.arff.txt'
        feature_types = [True for i in range(14)]
        feature_types.append(False)
        label_index = 14

    # german credit
    if False:
        dataset_name = 'german credit'
        ori_file = 'german.data.txt'
        feature_types = [False for i in range(21)]
        feature_types[1] = True
        feature_types[4] = True
        feature_types[7] = True
        feature_types[10] = True
        feature_types[12] = True
        feature_types[15] = True
        feature_types[17] = True
        label_index = 20

    # gisette use No.3 sub-processing
    if False:
        dataset_name = 'gisette'
        ori_train_data = 'gisette_train.data.txt'
        ori_train_label = 'gisette_train.labels.txt'
        ori_test_data = 'gisette_valid.data'
        ori_test_label = 'gisette_valid.labels'
        feature_types = [True for i in range(5000)]

    # jsbach, No.1
    if False:
        dataset_name = 'jsbach'
        ori_file = 'jsbach_chorals_harmony.data'
        feature_types = [False for i in range(17)]
        feature_types[15] = True
        label_index = 16

    # imageSegmentation_car, No.2
    if False:
        dataset_name = 'imageSegmentation_car'
        ori_train_file = 'segmentation.data.txt'
        ori_test_file = 'segmentation.test.txt'
        feature_types = [True for i in range(20)]
        feature_types[0] = False
        useless_index = []
        label_index = 0

    # iris, No.1
    if False:
        dataset_name = 'iris'
        ori_file = 'iris.data'
        feature_types = [True, True, True, True, False]
        label_index = 4

    # letterRecognition, No.1
    if False:
        dataset_name = 'letterRecognition'
        ori_file = 'letter-recognition.data'
        feature_types = [True for i in range(17)]
        feature_types[0] = False
        label_index = 0

    # madelon, No.3
    if False:
        dataset_name = 'madelon'
        ori_train_data = 'madelon_train.data.txt'
        ori_train_label = 'madelon_train.labels.txt'
        ori_test_data = 'madelon_valid.data.txt'
        ori_test_label = 'madelon_valid.labels.txt'
        feature_types = [True for i in range(500)]

    # magic04, No.1
    if False:
        dataset_name = 'magic04'
        ori_file = 'magic04.data.txt'
        feature_types = [True for i in range(11)]
        feature_types[10] = False
        label_index = 10

    # Diabetic Retinopathy Debrecen Data Set Data Set, No.1
    if False:
        dataset_name = 'messidor'
        ori_file = 'messidor_features.arff'
        feature_types = [True for i in range(20)]
        feature_types[19] = False
        label_index = 19

    # mushroom, No.1
    if False:
        dataset_name = 'mushroom'
        ori_file = 'agaricus-lepiota.data.txt'
        feature_types = [False for i in range(23)]
        label_index = 0

    # nursery, No.1
    if False:
        dataset_name = 'nursery'
        ori_file = 'nursery.data.txt'
        feature_types = [False for i in range(9)]
        label_index = 8

    # occupancy, No.4, delete 0-th and 1-st features
    if False:
        dataset_name = 'occupancy'
        ori_file = 'datatraining.txt'
        feature_types = [False, False, True, True, True, True, True, False]
        useless_index = [0, 1]
        label_index = 7

    # seismic, No.1
    if False:
        dataset_name = 'seismic'
        ori_file = 'seismic-bumps.arff.txt'
        feature_types = [True for i in range(19)]
        feature_types[0] = False
        feature_types[1] = False
        feature_types[2] = False
        feature_types[7] = False
        feature_types[18] = False
        label_index = 18

    # spambase, No.1
    if False:
        dataset_name = 'spambase'
        ori_file = 'spambase.data.txt'
        feature_types = [True for i in range(58)]
        feature_types[57] = False
        label_index = 57

    # statlogSegment, No.1
    if False:
        dataset_name = 'statlogSegment'
        ori_file = 'segment.data.txt'
        feature_types = [True for i in range(20)]
        feature_types[19] = False
        label_index = 19

    # wilt, No.5
    if False:
        dataset_name = 'wilt'
        ori_train_file = 'training.csv'
        ori_test_file = 'testing.csv'
        feature_types = [False, True, True, True, True, True]
        label_index = 0

    # wine_quality_red, No.1
    if False:
        dataset_name = 'wine_quality_red'
        ori_file = 'winequality-red.csv'
        feature_types = [True for i in range(12)]
        feature_types[11] = False
        label_index = 11

    # wine_quality_white, No.1
    if False:
        dataset_name = 'wine_quality_white'
        ori_file = 'winequality-white.csv'
        feature_types = [True for i in range(12)]
        feature_types[11] = False
        label_index = 11

    # yeast, no.4
    if False:
        dataset_name = 'yeast'
        ori_file = 'yeast.data.txt'
        feature_types = [False, True, True, True, True, True, True, True, True, False]
        useless_index = [0]
        label_index = 9

    # adult, No.6
    if False:
        dataset_name = 'adult'
        ori_train_file = 'adult.data'
        ori_test_file = 'adult.test'
        feature_types = [True, False, True, False, True, False, False, False, False, False, True, True, True,
                         False, False]
        label_index = 14

    # arcene, No.3
    if False:
        dataset_name = 'arcene'
        ori_train_data = 'arcene_train.data'
        ori_train_label = 'arcene_train.labels'
        ori_test_data = 'arcene_valid.data'
        ori_test_label = 'arcene_valid.labels'
        feature_types = [True for i in range(10000)]

    # breast_cancer_wisconsin, No.4
    if False:
        dataset_name = 'breast_cancer_wisconsin'
        ori_file = 'breast-cancer-wisconsin.data'
        feature_types = [False for i in range(11)]
        useless_index = [0]
        label_index = 10

    # covtype, No.1
    if False:
        dataset_name = 'covtype'
        ori_file = 'covtype.data'
        feature_types = [True for i in range(10)]
        for i in range(45):
            feature_types.append(False)
        label_index = 54

    # drug_consumption, No.1
    if False:
        dataset_name = 'drug_consumption'
        ori_file = 'drug_consumption.data'
        feature_types = [False]
        for i in range(12):
            feature_types.append(True)
        for i in range(19):
            feature_types.append(False)
        useless_index = [0]
        label_index = 31

    # ecoli, No.1
    if False:
        dataset_name = 'ecoli'
        ori_file = 'ecoli.data'
        feature_types = [False, True, True, True, True, True, True, True, False]
        label_index = 8

    # flags, No.4, choose religion as prediction target
    if False:
        dataset_name = 'flag'
        ori_file = 'flag.data'
        feature_types = [False for i in range(30)]
        feature_types[3] = True
        feature_types[4] = True
        for i in range(3):
            feature_types[7 + i] = True
        for i in range(5):
            feature_types[18 + i] = True
        useless_index = [0]
        label_index = 6

    # glass, No.4
    if False:
        dataset_name = 'glass'
        ori_file = 'glass.data'
        feature_types = [False]
        for i in range(9):
            feature_types.append(True)
        feature_types.append(False)
        useless_index = [0]
        label_index = 10

    # horse_colic, No.2
    if False:
        dataset_name = 'horse_colic'
        ori_train_file = 'horse-colic.data'
        ori_test_file = 'horse-colic.test'
        feature_types = [False for i in range(28)]
        feature_types[3] = True
        feature_types[4] = True
        feature_types[5] = True
        feature_types[15] = True
        feature_types[18] = True
        feature_types[19] = True
        feature_types[21] = True
        useless_index = [2]
        label_index = 23

    # HTRU2, No.1, line is split by '\r'
    if False:
        dataset_name = 'HTRU2'
        ori_file = 'HTRU_2.arff'
        feature_types = [True for i in range(8)]
        feature_types.append(False)
        label_index = 8

    # wdbc, No.4
    if False:
        dataset_name = 'wdbc'
        ori_file = 'wdbc.data'
        feature_types = [True for i in range(32)]
        feature_types[0] = False
        feature_types[1] = False
        useless_index = [0]
        label_index = 1

    # wpbc, No.4
    if False:
        dataset_name = 'wpbc'
        ori_file = 'wpbc.data'
        feature_types = [True for i in range(35)]
        feature_types[0] = False
        feature_types[1] = False
        useless_index = [0]
        label_index = 1

    if True:
        dataset_name = 'house_vote'
        ori_file = 'house-votes-84.data.txt'
        feature_types = [False for i in range(17)]
        useless_index = []
        label_index = 0

    # processing--------------------------------------------------------------
    # No.1
    if True:
        # just one data file, we should split training and testing data
        # cylinder, abalone, balanceScale, banknote, car, chess, chess2
        file_name = path + dataset_name + '/' + ori_file
        # reading data from file
        data = data_reader(file_name, feature_types)
        # missing value processing
        data = missing_value_processing(data, feature_types)
        # extract label
        data, label = extracting_label(data, label_index)
        del feature_types[label_index]
        # categorical feature encoding
        data = categorical_feature_encoding(data, feature_types)
        label, label_name = label_encoding(label)
        # split data into training and testing
        train_data, train_label, test_data, test_label = split_data(data, label, percent=0.2)

    # No.2
    if False:
        # training and testing data are in different files
        # anneal
        train_file_name = path + dataset_name + '/' + ori_train_file
        test_file_name = path + dataset_name + '/' + ori_test_file
        # get training and testing data
        train_data = data_reader(train_file_name, feature_types)
        test_data = data_reader(test_file_name, feature_types)
        # extract label from data
        train_data, train_label = extracting_label(train_data, label_index)
        test_data, test_label = extracting_label(test_data, label_index)
        del feature_types[label_index]
        # delete useless feature from data
        train_data = feature_selected(train_data, useless_index)
        test_data = feature_selected(test_data, useless_index)
        new_feature_types = []
        for i in range(len(feature_types)):
            if not (i in useless_index):
                new_feature_types.append(feature_types[i])
        feature_types = new_feature_types
        # processing training and testing data at the same time
        train_data_len = len(train_data)
        test_data_len = len(test_data)
        data = []
        data.extend(train_data)
        data.extend(test_data)
        label = []
        label.extend(train_label)
        label.extend(test_label)
        # missing feature processing
        data = missing_value_processing(data, feature_types)
        # encoding categorical feature
        data = categorical_feature_encoding(data, feature_types)
        # encoding label
        label, label_name = label_encoding(label)
        # regain training and testing data from data
        train_data = []
        train_label = []
        for i in range(train_data_len):
            train_data.append(data[0])
            train_label.append(label[0])
            del data[0]
            del label[0]
        test_data = data
        test_label = label

    # No.3
    if False:
        # training and testing data are in different files
        # gisette, madelon
        train_data_file_name = path + dataset_name + '/' + ori_train_data
        train_label_file_name = path + dataset_name + '/' + ori_train_label
        test_data_file_name = path + dataset_name + '/' + ori_test_data
        test_label_file_name = path + dataset_name + '/' + ori_test_label
        # get training and testing data
        train_data = data_reader(train_data_file_name, feature_types)
        train_label = data_reader(train_label_file_name, [False])
        train_label = np.array(train_label).reshape(len(train_label)).tolist()
        test_data = data_reader(test_data_file_name, feature_types)
        test_label = data_reader(test_label_file_name, [False])
        test_label = np.array(test_label).reshape(len(test_label)).tolist()
        # processing training and testing data at the same time
        train_data_len = len(train_data)
        print('training data length:', train_data_len)
        test_data_len = len(test_data)
        print('testing data length:', test_data_len)
        data = []
        data.extend(train_data)
        data.extend(test_data)
        label = []
        label.extend(train_label)
        label.extend(test_label)
        # missing feature processing
        data = missing_value_processing(data, feature_types)
        # encoding categorical feature
        data = categorical_feature_encoding(data, feature_types)
        # encoding label
        label, label_name = label_encoding(label)
        # regain training and testing data from data
        train_data = []
        train_label = []
        for i in range(train_data_len):
            train_data.append(data[0])
            train_label.append(label[0])
            del data[0]
            del label[0]
        test_data = data
        test_label = label

    # No.4
    # delete some feature based on No.1
    # occupancy, yeast
    if False:
        file_name = path + dataset_name + '/' + ori_file
        # reading data from file
        data = data_reader(file_name, feature_types)
        # missing value processing
        data = missing_value_processing(data, feature_types)
        # extract label
        data, label = extracting_label(data, label_index)
        del feature_types[label_index]
        # delete useless feature
        data = feature_selected(data, useless_index)
        new_feature_types = []
        for i in range(len(feature_types)):
            if not (i in useless_index):
                new_feature_types.append(feature_types[i])
        feature_types = new_feature_types
        # categorical feature encoding
        data = categorical_feature_encoding(data, feature_types)
        label, label_name = label_encoding(label)
        # split data into training and testing
        train_data, train_label, test_data, test_label = split_data(data, label, percent=0.2)

    # No.5
    # reading csv file
    if False:
        train_file_name = path + dataset_name + '/' + ori_train_file
        test_file_name = path + dataset_name + '/' + ori_test_file

        train_data = csv_data_reader(train_file_name, feature_types)
        test_data = csv_data_reader(test_file_name, feature_types)
        train_data, train_label = extracting_label(train_data, label_index)
        test_data, test_label = extracting_label(test_data, label_index)
        del feature_types[label_index]

        train_data_len = len(train_data)
        test_data_len = len(test_data)

        data = []
        label = []
        data.extend(train_data)
        data.extend(test_data)
        label.extend(train_label)
        label.extend(test_label)

        data = missing_value_processing(data, feature_types)

        data = categorical_feature_encoding(data, feature_types)

        label, label_name = label_encoding(label)

        # regain training and testing data from data
        train_data = []
        train_label = []
        for i in range(train_data_len):
            train_data.append(data[0])
            train_label.append(label[0])
            del data[0]
            del label[0]
        test_data = data
        test_label = label

    # No.6
    # training and testing data are in different files, processing based on No.1
    if False:
        train_file_name = path + dataset_name + '/' + ori_train_file
        test_file_name = path + dataset_name + '/' + ori_test_file

        train_data = data_reader(train_file_name, feature_types)
        test_data = data_reader(test_file_name, feature_types)
        train_data, train_label = extracting_label(train_data, label_index)
        test_data, test_label = extracting_label(test_data, label_index)
        del feature_types[label_index]

        train_data_len = len(train_data)
        test_data_len = len(test_data)

        data = []
        label = []
        data.extend(train_data)
        data.extend(test_data)
        label.extend(train_label)
        label.extend(test_label)

        data = missing_value_processing(data, feature_types)

        data = categorical_feature_encoding(data, feature_types)

        label, label_name = label_encoding(label)

        # regain training and testing data from data
        train_data = []
        train_label = []
        for i in range(train_data_len):
            train_data.append(data[0])
            train_label.append(label[0])
            del data[0]
            del label[0]
        test_data = data
        test_label = label

    print('-------------------------------------------------------------------------------------')
    print('training data length:', len(train_data), ', testing data length:', len(test_data))
    print('data feature size:', len(train_data[0]))
    print('-------------------------------------------------------------------------------------')
    train_data_file = path + dataset_name + '/' + dataset_name + '_train_data.pkl'
    test_data_file = path + dataset_name + '/' + dataset_name + '_test_data.pkl'
    label_class_file = path + dataset_name + '/' + dataset_name + '_label_class.txt'

    # writing train feature and label into pickle file
    print('writing training data...')
    f = open(train_data_file, 'wb')
    pickle.dump(train_data, f, 2)
    pickle.dump(train_label, f, 2)
    f.close()

    # writing test feature and label into pickle file
    print('writing testing data...')
    f = open(test_data_file, 'wb')
    pickle.dump(test_data, f, 2)
    pickle.dump(test_label, f, 2)
    f.close()

    # writing label class into txt file
    print('writing label class file...')
    buff = []
    buff.append(list2string(label_name))
    fo.FileWriter(label_class_file, buff, style='w')


if __name__ == '__main__':

    dataset_info()
    # dataset_processing()
    print('ccc')


