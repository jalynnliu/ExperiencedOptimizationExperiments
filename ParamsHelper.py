
from random import choice
import numpy as np

def sample_to_dict(eg_input,sample,label_coder):
    # sample x to dict
    result = {}
    index = 0
    for k, (type, obj) in eg_input.items():
        if str(type).__eq__('float') or str(type).__eq__("int"):
            result[k] = sample[index]
        else:
            result[k] = label_coder.decode(k, sample[index])
        index = index + 1
    return result

class Optimizer():

    def __init__(self,dim):
        self.dimension = dim
        pass

    def insert_sample(self):
        pass

    def single_step(self):
        result = []
        for row in self.dimension:
            result.append(choice(row))
        return result

class ParamsHelper():
    def __init__(self):
        self.encode_dict = {}
        self.decode_dict = {}
        self.index_dict = {}

    def add_param_dict(self,hyper_param_name,objs):
        index = 0
        array = []
        dict = {}
        for obj in objs:
            array.append(index)
            dict[index] = obj
            index = index + 1

        self.encode_dict[hyper_param_name] = array
        self.decode_dict[hyper_param_name] = dict

    def encode(self,type,index,key, objs):
        '''
        :param type: str/float/int
        :param index: 1/2/3/4/5..
        :param key: lr/max_depth/..
        :param objs: depends on situations
        :return:index,result,dimen_type: 0,[1,2],1
        '''
        self.index_dict[index] = key
        result = None
        dimen_type = 0   #  just reference of the dimmension set
        #### category to label ####
        if str(type).__eq__('float'):
            # for float type
            (lower, upper) = objs
            bound = [lower, upper]
            dimen_type = 0
        elif str(type).__eq__("int"):
            # for int type
            (lower, upper) = objs
            bound = [lower,upper]
            dimen_type = 1
        elif str(type).__eq__("grid"):
            self.add_param_dict(key, objs)
            result = self.encode_dict[key]
            dimen_type = 2
            bound = [result[0], result[-1]]
        else:
            #  for string type categorical
            self.add_param_dict(key,objs)
            result = self.encode_dict[key]
            dimen_type = 2
            bound = [result[0],result[-1]]
        return index,bound,dimen_type

    def decode(self,hyper_param_name,value):
        #### label to category ####
        if self.index_dict.__contains__(hyper_param_name):
            key = self.index_dict[hyper_param_name]
            if self.decode_dict.__contains__(key):
                return self.decode_dict[key][value]
            else:
                return value
        else:
            return value

    def sample_decode(self, x):
        '''
        get the real result output
        :param x: sample one step output
        :return:
        '''
        result = {}
        for i in range(len(x)):
            result[self.index_dict[i]] = self.decode(i,x[i])
        return result

    def safe_get_dict(self,dic,key):
        if dic.__contains__(key):
            return dic[key]
        else:
            return key