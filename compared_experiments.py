import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import classes
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import time
import pickle
import FileOperator as fo
from Tools import list2string

from SampleTrainingData import dataset_reader


def time_formulate(start_t, end_t):
    time_l = end_t - start_t
    if time_l < 0:
        print 'time error!'
        hour = 0
        minute = 0
        second = 0
    else:
        hour = int(time_l / 3600)
        time_l = time_l - hour*3600
        minute = int(time_l / 60)
        second = time_l - minute*60
    return hour, minute, second


def data_collector(index_set, X, Y):
    data = np.zeros((len(index_set), X.shape[1]))
    label = np.zeros(len(index_set))

    for i in xrange(index_set.shape[0]):
        data[i, :] = X[index_set[i], :]
        label[i] = Y[index_set[i]]
    return data, label


def validation_error(classifier, X, Y, k=5):

    kf = StratifiedKFold(n_splits=k, shuffle=False)

    error_list = []
    for train_index, test_index in kf.split(X, Y):

        train_feature, train_label = data_collector(train_index, X, Y)
        test_feature, test_label = data_collector(test_index, X, Y)

        classifier = classifier.fit(train_feature, train_label)

        prediction = classifier.predict(test_feature)

        acc = accuracy_score(test_label, prediction)

        error_list.append(1 - acc)

    ave_error = np.mean(np.array(error_list))

    return ave_error


def chosen_single_classifier(dataset_list):

    data_path = 'data_set/'
    logging_path = 'results/baseline/'

    repeat = 5

    for dataset_name in dataset_list:

        log_buffer = []
        print '========================================================'
        log_buffer.append('========================================================')
        print 'dataset: ', dataset_name
        log_buffer.append('dataset: ' + dataset_name)

        train_file = data_path + dataset_name + '/' + dataset_name + '_train_data.pkl'
        test_file = data_path + dataset_name + '/' + dataset_name + '_test_data.pkl'

        train_feature, train_label, test_feature, test_label = dataset_reader(train_file, test_file)

        print '     feature size:', train_feature.shape[0], ', feature dimension:', train_feature.shape[1]
        log_buffer.append('     feature size:' + str(train_feature.shape[0]) + ', feature dimension:' +
                          str(train_feature.shape[1]))
        print '     feature size:', test_feature.shape[0], ', feature dimension:', test_feature.shape[1]
        log_buffer.append('     feature size:' + str(test_feature.shape[0]) + ', feature dimension:' +
                          str(test_feature.shape[1]))

        dtc = DecisionTreeClassifier()
        mlpc = MLPClassifier()
        lr = LogisticRegression()
        svc = classes.SVC()
        gpc = GaussianProcessClassifier()
        pac = PassiveAggressiveClassifier()
        gnb = GaussianNB()
        sgdc = SGDClassifier()
        rfc = RandomForestClassifier()
        knn = KNeighborsClassifier()

        classifiers = [dtc, mlpc, lr, svc, gpc, pac, gnb, sgdc, rfc, knn]

        classifier_names = ['Decision Tree Classifier', 'MLPClassifier', 'LogisticRegression', 'SVC',
                            'Gaussian Process Classifier', 'Passive Aggressive Classifier', 'GaussianNB',
                            'SGDClassifier:', 'Random Forest Classifier', 'K-Neighbors Classifier']

        classifier_vali = []
        for c_i in xrange(len(classifiers)):

            classifier = classifiers[c_i]
            classifier_name = classifier_names[c_i]

            print '--------------------------------------------------------'
            log_buffer.append('--------------------------------------------------------')
            print classifier_name, ':'
            log_buffer.append(classifier_name + ':')
            start_t = time.time()
            vali_error = validation_error(classifier, train_feature, train_label, k=5)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append(
                '     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                ' seconds')
            classifier_vali.append(vali_error)
            print 'validation error:', vali_error

        min_index = classifier_vali.index(min(classifier_vali))
        best_c = classifiers[min_index]
        best_c_name = classifier_names[min_index]

        print 'test best============================================'
        print 'best c: ', best_c_name
        log_buffer.append('===================================')
        log_buffer.append('best classifier: ' + best_c_name)

        test_errors = []
        for r_i in xrange(repeat):
            print 'test repeat ', r_i, '-----------------------------'
            start_t = time.time()
            best_c = best_c.fit(train_feature, train_label)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                              ' seconds')
            predictions = best_c.predict(test_feature)
            accuracy = accuracy_score(test_label, predictions)
            print 'error: ', 1 - accuracy
            test_errors.append(1 - accuracy)

        log_buffer.append('errors: ' + list2string(test_errors))
        mean_error = np.mean(np.array(test_errors))
        print 'mean error: ', mean_error
        log_buffer.append('mean_error: ' + str(mean_error))

        logging_file = logging_path + dataset_name + '_chosen_single.txt'
        print dataset_name, ' logging...'
        fo.FileWriter(logging_file, log_buffer, 'w')


def single_classifier(dataset_list):

    data_path = 'data_set/'
    logging_path = 'results/baseline/'

    repeat = 5

    for dataset_name in dataset_list:

        log_buffer = []
        print '========================================================'
        log_buffer.append('========================================================')
        print 'dataset: ', dataset_name
        log_buffer.append('dataset: ' + dataset_name)

        train_file = data_path + dataset_name + '/' + dataset_name + '_train_data.pkl'
        test_file = data_path + dataset_name + '/' + dataset_name + '_test_data.pkl'

        train_feature, train_label, test_feature, test_label = dataset_reader(train_file, test_file)

        print '     feature size:', train_feature.shape[0], ', feature dimension:', train_feature.shape[1]
        log_buffer.append('     feature size:' + str(train_feature.shape[0]) + ', feature dimension:' +
                          str(train_feature.shape[1]))
        print '     feature size:', test_feature.shape[0], ', feature dimension:', test_feature.shape[1]
        log_buffer.append('     feature size:' + str(test_feature.shape[0]) + ', feature dimension:' +
                          str(test_feature.shape[1]))

        print '--------------------------------------------------------'
        log_buffer.append('--------------------------------------------------------')
        print 'Decision Tree Classifier:'
        log_buffer.append('Decision Tree Classifier:')
        error_list = []
        for rep_i in xrange(repeat):
            dtc = DecisionTreeClassifier()
            start_t = time.time()
            dtc = dtc.fit(train_feature, train_label)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                              ' seconds')
            predictions = dtc.predict(test_feature)
            accuracy = accuracy_score(test_label, predictions)
            error_list.append(1 - accuracy)
            print '     error rate: ',  1 - accuracy
            log_buffer.append('     error rate: ' + str(1 - accuracy))
        e_mean = np.mean(np.array(error_list))
        e_std = np.std(np.array(error_list))
        print 'mean error: ', e_mean, '#', e_std
        log_buffer.append('mean error: ' + str(e_mean) + '#' + str(e_std))

        print '--------------------------------------------------------'
        log_buffer.append('--------------------------------------------------------')
        print 'MLPClassifier:'
        log_buffer.append('MLPClassifier:')
        error_list = []
        for rep_i in xrange(repeat):
            mlpc = MLPClassifier()
            start_t = time.time()
            mlpc = mlpc.fit(train_feature, train_label)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                              ' seconds')
            predictions = mlpc.predict(test_feature)
            accuracy = accuracy_score(test_label, predictions)
            error_list.append(1 - accuracy)
            print '     error rate: ', 1 - accuracy
            log_buffer.append('     error rate: ' + str(1 - accuracy))
        e_mean = np.mean(np.array(error_list))
        e_std = np.std(np.array(error_list))
        print 'mean error: ', e_mean, '#', e_std
        log_buffer.append('mean error: ' + str(e_mean) + '#' + str(e_std))

        print '--------------------------------------------------------'
        log_buffer.append('--------------------------------------------------------')
        print 'LogisticRegression:'
        log_buffer.append('LogisticRegression:')
        error_list = []
        for rep_i in xrange(repeat):
            lr = LogisticRegression()
            start_t = time.time()
            lr = lr.fit(train_feature, train_label)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                              ' seconds')
            predictions = lr.predict(test_feature)
            accuracy = accuracy_score(test_label, predictions)
            error_list.append(1 - accuracy)
            print '     error rate: ', 1 - accuracy
            log_buffer.append('     error rate: ' + str(1 - accuracy))
        e_mean = np.mean(np.array(error_list))
        e_std = np.std(np.array(error_list))
        print 'mean error: ', e_mean, '#', e_std
        log_buffer.append('mean error: ' + str(e_mean) + '#' + str(e_std))

        print '--------------------------------------------------------'
        log_buffer.append('--------------------------------------------------------')
        print 'SVC:'
        log_buffer.append('SVC:')
        error_list = []
        for rep_i in xrange(repeat):
            svc = classes.SVC()
            start_t = time.time()
            svc = svc.fit(train_feature, train_label)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                              ' seconds')
            predictions = svc.predict(test_feature)
            accuracy = accuracy_score(test_label, predictions)
            error_list.append(1 - accuracy)
            print '     error rate: ', 1 - accuracy
            log_buffer.append('     error rate: ' + str(1 - accuracy))
        e_mean = np.mean(np.array(error_list))
        e_std = np.std(np.array(error_list))
        print 'mean error: ', e_mean, '#', e_std
        log_buffer.append('mean error: ' + str(e_mean) + '#' + str(e_std))

        print '--------------------------------------------------------'
        log_buffer.append('--------------------------------------------------------')
        print 'Gaussian Process Classifier:'
        log_buffer.append('Gaussian Process Classifier:')
        error_list = []
        for rep_i in xrange(repeat):
            gpc = GaussianProcessClassifier()
            start_t = time.time()
            gpc = gpc.fit(train_feature, train_label)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                              ' seconds')
            predictions = gpc.predict(test_feature)
            accuracy = accuracy_score(test_label, predictions)
            error_list.append(1 - accuracy)
            print '     error rate: ', 1 - accuracy
            log_buffer.append('     error rate: ' + str(1 - accuracy))
        e_mean = np.mean(np.array(error_list))
        e_std = np.std(np.array(error_list))
        print 'mean error: ', e_mean, '#', e_std
        log_buffer.append('mean error: ' + str(e_mean) + '#' + str(e_std))

        print '--------------------------------------------------------'
        log_buffer.append('--------------------------------------------------------')
        print 'Passive Aggressive Classifier:'
        log_buffer.append('Passive Aggressive Classifier:')
        error_list = []
        for rep_i in xrange(repeat):
            pac = PassiveAggressiveClassifier()
            start_t = time.time()
            pac = pac.fit(train_feature, train_label)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                              ' seconds')
            predictions = pac.predict(test_feature)
            accuracy = accuracy_score(test_label, predictions)
            error_list.append(1 - accuracy)
            print '     error rate: ', 1 - accuracy
            log_buffer.append('     error rate: ' + str(1 - accuracy))
        e_mean = np.mean(np.array(error_list))
        e_std = np.std(np.array(error_list))
        print 'mean error: ', e_mean, '#', e_std
        log_buffer.append('mean error: ' + str(e_mean) + '#' + str(e_std))

        print '--------------------------------------------------------'
        log_buffer.append('--------------------------------------------------------')
        print 'GaussianNB:'
        log_buffer.append('GaussianNB:')
        error_list = []
        for rep_i in xrange(repeat):
            gnb = GaussianNB()
            start_t = time.time()
            gnb = gnb.fit(train_feature, train_label)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                              ' seconds')
            predictions = gnb.predict(test_feature)
            accuracy = accuracy_score(test_label, predictions)
            error_list.append(1 - accuracy)
            print '     error rate: ', 1 - accuracy
            log_buffer.append('     error rate: ' + str(1 - accuracy))
        e_mean = np.mean(np.array(error_list))
        e_std = np.std(np.array(error_list))
        print 'mean error: ', e_mean, '#', e_std
        log_buffer.append('mean error: ' + str(e_mean) + '#' + str(e_std))

        print '--------------------------------------------------------'
        log_buffer.append('--------------------------------------------------------')
        print 'SGDClassifier:'
        log_buffer.append('SGDClassifier:')
        error_list = []
        for rep_i in xrange(repeat):
            sgdc = SGDClassifier()
            start_t = time.time()
            sgdc = sgdc.fit(train_feature, train_label)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                              ' seconds')
            predictions = sgdc.predict(test_feature)
            accuracy = accuracy_score(test_label, predictions)
            error_list.append(1 - accuracy)
            print '     error rate: ', 1 - accuracy
            log_buffer.append('     error rate: ' + str(1 - accuracy))
        e_mean = np.mean(np.array(error_list))
        e_std = np.std(np.array(error_list))
        print 'mean error: ', e_mean, '#', e_std
        log_buffer.append('mean error: ' + str(e_mean) + '#' + str(e_std))

        print '--------------------------------------------------------'
        log_buffer.append('--------------------------------------------------------')
        print 'Random Forest Classifier:'
        log_buffer.append('Random Forest Classifier:')
        error_list = []
        for rep_i in xrange(repeat):
            rfc = RandomForestClassifier()
            start_t = time.time()
            rfc = rfc.fit(train_feature, train_label)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                              ' seconds')
            predictions = rfc.predict(test_feature)
            accuracy = accuracy_score(test_label, predictions)
            error_list.append(1 - accuracy)
            print '     error rate: ', 1 - accuracy
            log_buffer.append('     error rate: ' + str(1 - accuracy))
        e_mean = np.mean(np.array(error_list))
        e_std = np.std(np.array(error_list))
        print 'mean error: ', e_mean, '#', e_std
        log_buffer.append('mean error: ' + str(e_mean) + '#' + str(e_std))

        print '--------------------------------------------------------'
        log_buffer.append('--------------------------------------------------------')
        print 'K-Neighbors Classifier:'
        log_buffer.append('K-Neighbors Classifier:')
        error_list = []
        for rep_i in xrange(repeat):
            knn = KNeighborsClassifier()
            start_t = time.time()
            knn = knn.fit(train_feature, train_label)
            end_t = time.time()
            hour, minute, second = time_formulate(start_t, end_t)
            print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
            log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                              ' seconds')
            predictions = knn.predict(test_feature)
            accuracy = accuracy_score(test_label, predictions)
            error_list.append(1 - accuracy)
            print '     error rate: ', 1 - accuracy
            log_buffer.append('     error rate: ' + str(1 - accuracy))
        e_mean = np.mean(np.array(error_list))
        e_std = np.std(np.array(error_list))
        print 'mean error: ', e_mean, '#', e_std
        log_buffer.append('mean error: ' + str(e_mean) + '#' + str(e_std))

        logging_file = logging_path + dataset_name + '_baseline.txt'
        print dataset_name, ' logging...'
        fo.FileWriter(logging_file, log_buffer, 'w')



def default_ensemble_classifier(dataset_list):

    data_path = 'data_set/'
    logging_path = 'results/default_ec/'

    for dataset_name in dataset_list:
        log_buffer = []
        print '========================================================'
        log_buffer.append('========================================================')
        print 'dataset: ', dataset_name
        log_buffer.append('dataset: ' + dataset_name)

        train_file = data_path + dataset_name + '/' + dataset_name + '_train_data.pkl'
        test_file = data_path + dataset_name + '/' + dataset_name + '_test_data.pkl'

        train_feature, train_label, test_feature, test_label = dataset_reader(train_file, test_file)

        dtc = DecisionTreeClassifier()
        mlpc = MLPClassifier()
        lr = LogisticRegression()
        svc = classes.SVC()
        gpc = GaussianProcessClassifier()
        pac = PassiveAggressiveClassifier()
        gnb = GaussianNB()
        sgdc = SGDClassifier()
        rfc = RandomForestClassifier()
        knn = KNeighborsClassifier()

        weight = [0.1 for i in xrange(10)]

        estimators = [('dtc', dtc), ('mlpc', mlpc), ('lr', lr), ('svc', svc), ('gpc', gpc), ('pac', pac),
                      ('gnb', gnb), ('sgdc', sgdc), ('rfc', rfc), ('knn', knn)]

        voting = VotingClassifier(estimators, voting='hard', weights=weight, n_jobs=-1)

        print 'default ensemble classifier:'
        log_buffer.append('default eensemble classifier:')

        start_t = time.time()
        voting = voting.fit(train_feature, train_label)
        end_t = time.time()

        hour, minute, second = time_formulate(start_t, end_t)
        print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
        log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                          ' seconds')

        predictions = voting.predict(test_feature)
        accuracy = accuracy_score(test_label, predictions)
        print '     error rate: ', 1 - accuracy
        log_buffer.append('     error rate: ' + str(1 - accuracy))

        logging_file = logging_path + dataset_name + 'default_ec.txt'
        print dataset_name, ' logging...'
        fo.FileWriter(logging_file, log_buffer, 'w')


def default_ensemble_classifier2(dataset_list):

    data_path = 'data_set/'
    logging_path = 'results/default_ec2/'

    for dataset_name in dataset_list:
        log_buffer = []
        print '========================================================'
        log_buffer.append('========================================================')
        print 'dataset: ', dataset_name
        log_buffer.append('dataset: ' + dataset_name)

        train_file = data_path + dataset_name + '/' + dataset_name + '_train_data.pkl'
        test_file = data_path + dataset_name + '/' + dataset_name + '_test_data.pkl'

        train_feature, train_label, test_feature, test_label = dataset_reader(train_file, test_file)

        dtc = DecisionTreeClassifier()
        mlpc = MLPClassifier()
        lr = LogisticRegression()
        svc = classes.SVC()
        gpc = GaussianProcessClassifier()
        gnb = GaussianNB()
        rfc = RandomForestClassifier()
        knn = KNeighborsClassifier()

        weight = [0.125 for i in xrange(8)]

        estimators = [('dtc', dtc), ('mlpc', mlpc), ('lr', lr), ('svc', svc), ('gpc', gpc),
                      ('gnb', gnb), ('rfc', rfc), ('knn', knn)]

        voting = VotingClassifier(estimators, voting='hard', weights=weight, n_jobs=-1)

        print 'default ensemble classifier:'
        log_buffer.append('default eensemble classifier:')

        start_t = time.time()
        voting = voting.fit(train_feature, train_label)
        end_t = time.time()

        hour, minute, second = time_formulate(start_t, end_t)
        print '     training time: ', hour, ' hours, ', minute, ' minutes, ', second, ' seconds'
        log_buffer.append('     training time: ' + str(hour) + ' hours, ' + str(minute) + ' minutes, ' + str(second) +
                          ' seconds')

        predictions = voting.predict(test_feature)
        accuracy = accuracy_score(test_label, predictions)
        print '     error rate: ', 1 - accuracy
        log_buffer.append('     error rate: ' + str(1 - accuracy))

        logging_file = logging_path + dataset_name + 'default_ec.txt'
        print dataset_name, ' logging...'
        fo.FileWriter(logging_file, log_buffer, 'w')


def lalala(ranks):

    for rank in ranks:
        print '-------------------------------'
        print 'rank: ', rank

        sum_r = 0.0
        for i in xrange(len(rank)):
            sum_r += (i+1) * rank[i]

        total = sum(rank)

        print 'avg rank: ', float(sum_r) / total







if __name__ == '__main__':
    # dataset_list = ['annealing', 'arcene', 'balanceScale', 'banknote', 'breast_cancer_wisconsin', 'car',
    #                 'chess', 'cmc', 'CNAE9', 'credit', 'cylinder', 'drug_consumption', 'ecoli', 'flag', 'german credit',
    #                 'glass', 'horse_colic', 'imageSegmentation_car', 'iris', 'madelon', 'messidor', 'mushroom',
    #                 'occupancy', 'seismic', 'spambase', 'statlogSegment', 'wdbc', 'wilt', 'wine_quality_red', 'wpbc',
    #                 'yeast', 'house_vote']
    #
    # default_ensemble_classifier2(dataset_list)

    ranks = [[15, 8, 1, 0, 0], [8, 10, 4, 2, 0], [1, 1, 6, 8, 8], [0, 2, 9, 9, 4], [4, 1, 4, 6, 9],
             [15, 5, 4, 0, 0], [5, 5, 5, 4, 5], [4, 3, 7, 4, 6], [3, 4, 7, 7, 3], [8, 5, 4, 3, 4],
             [10, 0, 0, 0, 0], [1, 7, 2, 0, 0], [0, 0, 4, 6, 0], [0, 0, 3, 5, 2], [0, 2, 1, 0, 7],
             [6, 4, 0, 0, 0], [1, 3, 3, 1, 2], [0, 1, 3, 2, 4], [1, 1, 2, 4, 2], [2, 1, 3, 2, 2]]

    lalala(ranks)