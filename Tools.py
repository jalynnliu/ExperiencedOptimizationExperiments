'''
Some tools were implemented in this files

Author:
    Yi-Qi Hu

Time:
    2016.6.13
'''

'''
 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

 Copyright (C) 2015 Nanjing University, Nanjing, China
'''

import random
import csv
import os

import numpy as np
import psutil
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing.label import _encode_python, _encode_numpy
from sklearn.utils.validation import check_is_fitted, column_or_1d, _num_samples

path = '/data/ExpAdaptation/RealProbsData/'


# used for generating number randomly
class RandomOperator:

    def __init__(self):
        return

    def get_uniform_integer(self, lower, upper):
        return random.randint(lower, upper)

    def get_uniform_double(self, lower, upper):
        return random.uniform(lower, upper)


class NumberSelector(BaseEstimator, TransformerMixin):
    """
     Feature selector acoording the feature index of the raw dataset
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        r_num, _ = X.shape
        return X[:, self.key].reshape((r_num, -1))


class IntergerTransform(BaseEstimator, TransformerMixin):
    """
     Feature selector acoording the feature index of the raw dataset
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        r_num, _ = X.shape
        return X[:].astype(np.float)


def list2string(list):
    my_str = str(list[0])
    i = 1
    while i < len(list):
        my_str = my_str + ' ' + str(list[i])
        i += 1
    return my_str


def _encode(values, uniques=None, encode=False):
    if values.dtype == object:
        return _encode_python(values, uniques, encode)
    else:
        return _encode_numpy(values, uniques, encode)


def get_standard_encoding(attribut):
    '''
    get selector accorting the attribute define here
    :param keyname:
    :param attribut:
    :return:
    '''
    if attribut == 'numerical':
        return IntergerTransform()
    else:
        return MutiLabelEncoder()


def string2list(string):
    string = string.split(' ')
    this_list = [float(string[i]) for i in range(len(string))]
    return this_list


class MutiLabelEncoder(BaseEstimator, TransformerMixin):
    '''
    multilabe encoder based on the labelEncoder
    '''

    def fit(self, y, *args, **kwargs):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = _encode(y)
        return self

    def fit_transform(self, y, *args, **kwargs):
        """Fit label encoder and return encoded labels

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        y = column_or_1d(y, warn=True)
        self.classes_, y = _encode(y, encode=True)
        return y.reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])
        self.classes_ = _encode(y)
        _, y = _encode(y, uniques=self.classes_, encode=True)
        return y.reshape(-1, 1)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)
        # inverse transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if len(diff):
            raise ValueError(
                "y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        return self.classes_[y]


class BenchmarkHelper():
    def __init__(self, check_nan=True):
        self.check_nan = check_nan
        self.models = []
        self.metrics = []
        pid = os.getpid()
        self.ps = psutil.Process(pid)

    def dic_to_array(self, dict):
        results = []
        for k, v in dict.items():
            results.append((k, v))
        return results

    def read_csv(self, dest_file):
        '''
        read csv and throw the nan data
        :param dest_file:
        :return:
        '''

        def is_list_contain_nan(list):
            for item in list:
                if str(item) == 'nan':
                    return True
            return False

        with open(dest_file, 'r') as dest_f:
            data_iter = csv.reader(dest_f, delimiter=',', quotechar='"')
            data = []
            for iter in data_iter:
                if self.check_nan and is_list_contain_nan(iter):
                    # is need to check nan and iter has nan then just regard of this row data
                    continue
                data.append(iter)
            # data = [data for data in data_iter]
        data_array = np.asarray(data, dtype=None)
        return data_array

    def get_features_description(self, feature_dir):
        '''
        produce the feature description according to the features.csv file
        :param feature_dir:
        :return:
        '''
        self.features = self.read_csv(feature_dir)
        description = '## Brief Description of %s Dataset\n\n' % self.dataset_name
        description = description + '#Number of TRAIN samples : %s\n' % str(self.train_X.shape[0])
        description = description + '#Number of TEST samples : %s\n' % str(self.test_X.shape[0])
        description = description + '#Number of features:  %s\n' % str(self.features.shape[0])
        description = description + "| feature_name  | feature type1 | feature type2    | item_count   |\n"
        description = description + "|:----:|:----:|:----:|:----:|\n"
        for col in self.features[:]:
            description = description + "| {}  | {} | {}     | {} |\n".format(col[0], col[1], col[2], col[3])
        print("Dataset {} contains features:".format(self.dataset_name))
        print(self.features[:, 0])
        return description

    def fetch_data(self, dataset_name):
        '''
        split the dataset
        :param dataset_name:
        :return:
        '''
        self.dataset_name = dataset_name
        train_dir = path + dataset_name + '/opt_train.csv'
        test_dir = path + dataset_name + '/test.csv'
        val_dir = path + dataset_name + '/opt_val.csv'

        train_data = self.read_csv(train_dir)
        test_data = self.read_csv(test_dir)
        val_data = self.read_csv(val_dir)
        assert len(train_data) > 0, "train data must be greater than zero"
        self.train_X, self.test_X, self.train_Y, self.test_Y, self.val_X, self.val_y = train_data[:, 0:-1], test_data[:,
                                                                                                            0:-1], \
                                                                                       train_data[:, -1], test_data[:,
                                                                                                          -1], \
                                                                                       val_data[:, :-1], val_data[:, -1]

        return self.train_X, self.test_X, self.train_Y, self.test_Y, self.val_X, self.val_y

    ## remove the empty array in the features array
    def remove_empty_str_array(self, features):
        result = []
        for feature in features:
            if len(feature) != 0:
                result.append(feature)
        return result

    def get_train_test_data(self, dataset_name, miss_fun_hdl=None):
        '''
        :param dataset_name: datasets name
        :param features:  the features in the datasets
        :param miss_fun_hdl: the function used to handler missing value
        :return:
        '''
        feature_dir = path + dataset_name + '/features.csv'
        self.features = self.read_csv(feature_dir)
        features = self.remove_empty_str_array(self.features)
        train_X, test_X, train_Y, test_Y, val_X, val_Y = self.fetch_data(dataset_name)
        ## here is code handling miss value
        if miss_fun_hdl is not None:
            train_X, test_X = miss_fun_hdl(train_X, test_X)
        feature_unions, header = self.get_default_feature_unions_pipeline(features)
        label_pipeline = self.get_label_pipeline()
        train_Y = self.get_transformed_data(label_pipeline, train_Y.reshape((-1, 1)))
        test_Y = self.get_transformed_data(label_pipeline, test_Y.reshape((-1, 1)))
        val_Y = self.get_transformed_data(label_pipeline, val_Y.reshape((-1, 1)))
        train_X = self.get_transformed_data(feature_unions, train_X)
        test_X = self.get_transformed_data(feature_unions, test_X)
        val_X = self.get_transformed_data(feature_unions, val_X)
        self.train = np.concatenate((train_X, train_Y), axis=1)
        self.test = np.concatenate((test_X, test_Y), axis=1)
        self.val = np.concatenate((val_X, val_Y), axis=1)
        return self.train, self.test, self.val

    def get_number_selector(self, feature_names):
        def get_key_by_feature_name(feature_name):
            keyobj = np.where(self.features[:, 0] == feature_name)
            assert len(keyobj[0]) > 0, "feature name must be defined"
            return keyobj[0][0]

        keys = []
        if isinstance(feature_names, list):
            for feature_name in feature_names:
                keys.append(get_key_by_feature_name(feature_name))
        else:
            keys.append(get_key_by_feature_name(feature_names))

        # iterate every column in the feature dataset
        select = NumberSelector(keys)
        return select

    # pipeline default maker
    def get_default_feature_unions_pipeline(self, features=[]):
        '''
        generate the default feature unions according the feature type
        :param features:
        :return: feature unions dict  {'feature1':....,'feature2':...,}
        '''
        # feature_dir = "./cache/{}/features.csv".format(self.dataset_name)
        feature_data = self.features
        feature_unions = {}
        features_transform = []
        for col in feature_data[0:-1]:
            keyname = col[0]
            # key_attri = {categorical numerical}
            key_attri = col[1]
            if keyname not in feature_data and len(features) != 0:
                continue
            features_transform.append(keyname)
            selct = self.get_number_selector(keyname)
            feature_unions[keyname] = Pipeline([
                ('selector', selct),
                # ('standard2',get_standard_encoding2(key_attri)),
                ('standard', get_standard_encoding(key_attri))
            ])
        return feature_unions, features_transform

    def get_label_pipeline(self):
        feature_data = self.features
        col = feature_data[-1]
        keyname = col[0]
        feature_unions = {}
        feature_unions[keyname] = Pipeline([
            ('selector', NumberSelector(0)),
            ('standard', MutiLabelEncoder())
        ])
        return feature_unions

    def get_transformed_data(self, features_union, raw_data):
        assert len(features_union) > 0, "the number of feature pipelines must more than 0"
        unions = FeatureUnion(self.dic_to_array(features_union))
        feature_processing = Pipeline([('unions', unions)])
        return feature_processing.fit_transform(raw_data)
