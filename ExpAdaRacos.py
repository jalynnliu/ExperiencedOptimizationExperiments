from Components import Instance
from Components import Dimension
from Tools import RandomOperator
from Tools import list2string
import numpy as np
import torch
from torch.autograd import Variable
import math
import copy
import matplotlib.pyplot as plt

step = 100


class Experts(object):

    def __init__(self, predictors=None, eta=0.9, bg=50):

        self.predictors = predictors
        self.weights = []
        self.eta = eta
        self.pic_count = 0
        self.bg = bg
        return

    # prediction for each input
    # return: w_probs - weighted sum predictions of predictors
    #         outputs - prediction matrix for inputs
    def predict(self, inputs):

        inputs = torch.from_numpy(np.array(inputs)).float()
        inputs = Variable(inputs.cuda())

        # print 'input: ', inputs
        # inputs = Variable(inputs)

        outputs = []
        y = []

        for i in range(len(self.predictors)):
            # print i, ' predictor---'
            output = self.predictors[i].predictor(inputs)
            # print 'out 1: ', output
            output = output.data.cpu().numpy()
            # print 'out numpy: ', output
            output = output.reshape(output.size).tolist()
            # print 'out list: ', output

            y.append(output)
            if (i + 1) % step == 0:
                outputs.append(np.mean(np.array(y), axis=0))
                y = []
        # outputs = np.array(outputs)
        # print 'all out: ', outputs
        # outputs = np.mean(outputs, axis=0).tolist()
        # print 'mean out: ', outputs
        # print '-------------------------------'
        weight_sum = sum(self.weights)
        this_weights = np.array(copy.deepcopy(self.weights)) / weight_sum
        w_probs = np.array(outputs).T.dot(np.array(this_weights).T).tolist()
        return w_probs, outputs

    def update_weights(self, predictions, label, flag):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] * math.exp(-self.eta * self.loss_function(predictions[i], label))

        x = np.array(self.weights)
        # self.weights-=min(x)
        self.weights /= x.sum()
        # self.weights = self.weights.tolist()
        if (flag and self.pic_count < (self.bg / 10)) or self.pic_count == 0:
            if False:
                index = [i * 2 + 1 for i in range(int(len(self.weights) / 2))]
                plt.scatter(range(len(self.weights[index])), self.weights[index], c='red')
                plt.scatter(range(len(self.weights[[i * 2 for i in range(int(len(self.weights) / 2))]])),
                            self.weights[[i * 2 for i in range(int(len(self.weights) / 2))]], c='blue')
            else:
                plt.scatter(range(len(self.weights)), self.weights)
            plt.show()
            self.pic_count += 1
            print(self.weights)

        return

    def loss_function(self, prediction, label):
        return (label - prediction) * (label - prediction)

    def reset_weight(self):
        self.weights = [step / len(self.predictors) for _ in range(int(len(self.predictors) / step))]
        return

    def delete_predictors(self):

        return




class ExpAdaRacosOptimization:

    def __init__(self, dimension, expert):

        self.__pop = []  # population set
        self.__pos_pop = []  # positive sample set
        self.__optimal = None  # the best sample so far
        self.__region = []  # the region of model
        self.__label = []  # the random label, if true random in this dimension
        self.__sample_size = 0  # the instance size of sampling in an iteration
        self.__budget = 0  # budget of evaluation
        self.__positive_num = 0  # positive sample set size
        self.__rand_probability = 0.0  # the probability of sampling in model
        self.__uncertain_bit = 0  # the dimension size of sampling randomly
        self.__dimension = dimension

        # sampling saving
        self.__model_ins = []  # positive sample used to modeling
        self.__negative_set = []  # negative set used to modeling
        self.__new_ins = []  # new sample
        self.__sample_label = []  # the label of each sample

        self.__expert = expert
        self.__adv_threshold = 0

        self.__sample_results = []

        self.__optimal_update_count = 0

        for i in range(self.__dimension.get_size()):
            region = [0.0, 0.0]
            self.__region.append(region)
            self.__label.append(0)

        self.__ro = RandomOperator()
        return

    # --------------------------------------------------
    # debugging
    # print positive set
    def show_pos_pop(self):
        print('positive set:------------------')
        for i in range(self.__positive_num):
            self.__pos_pop[i].show_instance()
        print('-------------------------------')
        return

    # print negative set
    def show_pop(self):
        print('negative set:------------------')
        for i in range(self.__sample_size):
            self.__pop[i].show_instance()
        print('-------------------------------')
        return

    # print optimal
    def show_optimal(self):
        print('optimal:-----------------------')
        self.__optimal.show_instance()
        print('-------------------------------')

    # print region
    def show_region(self):
        print('region:------------------------')
        for i in range(self.__dimension.get_size()):
            print('dimension ', i, ' [', self.__region[i][0], ',', self.__region[i][1], ']')
        print('-------------------------------')
        return

    # print label
    def show_label(self):
        print('label:-------------------------')
        print(self.__label)
        print('-------------------------------')

    # --------------------------------------------------

    def get_predictors(self):
        return self.__predictors

    # clear sampling log
    def log_clear(self):
        self.__model_ins = []
        self.__negative_set = []
        self.__new_ins = []
        self.__sample_label = []
        return

    # generate environment that should be logged
    # return is 2-d list
    def generate_environment(self, ins, new_ins):

        trajectory = []
        if True:
            array_best = []
            for i in range(len(ins.get_features())):
                array_best.append(ins.get_features()[i])
            new_sam = []
            for i in range(len(new_ins.get_features())):
                new_sam.append(new_ins.get_features()[i])
            sorted_neg = sorted(self.__pop, key=lambda instance: instance.get_fitness())
            for i in range(self.__sample_size):
                trajectory.append(sorted_neg[i].get_features())

        return array_best, trajectory, new_sam

    # clear parameters of saving
    def clear(self):
        self.__pop = []
        self.__pos_pop = []
        self.__optimal = None
        self.__expert.reset_weight()
        return

    # parameters setting
    def set_parameters(self, ss=0, bud=0, pn=0, rp=0.0, ub=0, at=0):
        self.__sample_size = ss
        self.__budget = bud
        self.__positive_num = pn
        self.__rand_probability = rp
        self.__uncertain_bit = ub
        self.__adv_threshold = at
        return

    # get optimal
    def get_optimal(self):
        return self.__optimal

    # generate an instance randomly
    def random_instance(self, dim, region, label):
        ins = Instance(dim)
        for i in range(dim.get_size()):
            if label[i] is True:
                if dim.get_type(i) == 0:
                    ins.set_feature(i, self.__ro.get_uniform_double(region[i][0], region[i][1]))
                else:
                    ins.set_feature(i, self.__ro.get_uniform_integer(region[i][0], region[i][1]))
        return ins

    # generate an instance based on a positive sample
    def pos_random_instance(self, dim, region, label, pos_instance):
        ins = Instance(dim)
        for i in range(dim.get_size()):
            if label[i] is False:
                if dim.get_type(i) == 0:
                    ins.set_feature(i, self.__ro.get_uniform_double(region[i][0], region[i][1]))
                else:
                    ins.set_feature(i, self.__ro.get_uniform_integer(region[i][0], region[i][1]))
            else:
                ins.set_feature(i, pos_instance.get_feature(i))
        return ins

    # reset model
    def reset_model(self):
        for i in range(self.__dimension.get_size()):
            self.__region[i][0] = self.__dimension.get_region(i)[0]
            self.__region[i][1] = self.__dimension.get_region(i)[1]
            self.__label[i] = True
        return

    # if an instance exists in a list, return true
    def instance_in_list(self, ins, this_list, end):
        for i in range(len(this_list)):
            if i == end:
                break
            if ins.equal(this_list[i]) is True:
                return True
        return False

    # initialize pop, pos_pop, optimal
    def initialize(self, func):

        temp = []

        self.reset_model()

        # sample in original region under uniform distribution
        for i in range(self.__sample_size + self.__positive_num):
            while True:
                ins = self.random_instance(self.__dimension, self.__region, self.__label)
                if self.instance_in_list(ins, temp, i) is False:
                    break
            ins.set_fitness(func(ins.get_features()))
            self.__sample_results.append(ins.get_fitness())
            temp.append(ins)

            # sorted by fitness
            temp.sort(key=lambda instance: instance.get_fitness())

        # initialize pos_pop
        for i in range(self.__positive_num):
            self.__pos_pop.append(temp[i])

        # initialize pop
        for i in range(self.__sample_size):
            self.__pop.append(temp[self.__positive_num + i])

        # initialize optimal
        self.__optimal = self.__pos_pop[0].copy_instance()

        return

    # distinguish function for mixed optimization
    def distinguish(self, exa, neg_set):
        for i in range(self.__sample_size):
            j = 0
            while j < self.__dimension.get_size():
                if self.__dimension.get_type(j) == 0 or self.__dimension.get_type(j) == 1:
                    if neg_set[i].get_feature(j) < self.__region[j][0] or neg_set[i].get_feature(j) > self.__region[j][
                        1]:
                        break
                else:
                    if self.__label[j] is False and exa.get_feature(j) != neg_set[i].get_feature(j):
                        break
                j += 1
            if j >= self.__dimension.get_size():
                return False
        return True

    # update positive set and negative set using a new sample by online strategy
    def online_update(self, ins):

        # update positive set
        j = 0
        while j < self.__positive_num:
            if ins.get_fitness() < self.__pos_pop[j].get_fitness():
                break
            else:
                j += 1

        if j < self.__positive_num:
            temp = ins
            ins = self.__pos_pop[self.__positive_num - 1]
            k = self.__positive_num - 1
            while k > j:
                self.__pos_pop[k] = self.__pos_pop[k - 1]
                k -= 1
            self.__pos_pop[j] = temp

        # update negative set
        j = 0
        while j < self.__sample_size:
            if ins.get_fitness() < self.__pop[j].get_fitness():
                break
            else:
                j += 1
        if j < self.__sample_size:
            temp = ins
            ins = self.__pop[self.__sample_size - 1]
            k = self.__sample_size - 1
            while k > j:
                self.__pop[k] = self.__pop[k - 1]
                k -= 1
            self.__pop[j] = temp

        return

    # update optimal
    def update_optimal(self):
        if self.__pos_pop[0].get_fitness() < self.__optimal.get_fitness():
            self.__optimal_update_count += 1
            self.__optimal = self.__pos_pop[0].copy_instance()
        return

    # generate instance randomly based on positive sample for mixed optimization
    def pos_random_mix_isntance(self, exa, region, label):
        ins = Instance(self.__dimension)
        for i in range(self.__dimension.get_size()):
            if label[i] is False:
                ins.set_feature(i, exa.get_feature(i))
            else:
                # float random
                if self.__dimension.get_type(i) == 0:
                    ins.set_feature(i, self.__ro.get_uniform_double(region[i][0], region[i][1]))
                # integer random
                elif self.__dimension.get_type(i) == 1:
                    ins.set_feature(i, self.__ro.get_uniform_integer(region[i][0], region[i][1]))
                # categorical random
                else:
                    ins.set_feature(i, self.__ro.get_uniform_integer(self.__dimension.get_region(i)[0],
                                                                     self.__dimension.get_region(i)[1]))
        return ins

    # generate model based on mixed optimization for next sampling
    # label[i] = false means this dimension should be set as the value in same dimension of positive sample
    def shrink_model(self, exa, neg_set):

        dist_count = 0

        chosen_dim = []
        non_chosen_dim = [i for i in range(self.__dimension.get_size())]

        remain_neg = [i for i in range(self.__sample_size)]

        while len(remain_neg) != 0:

            dist_count += 1

            temp_dim = non_chosen_dim[self.__ro.get_uniform_integer(0, len(non_chosen_dim) - 1)]
            chosen_neg = self.__ro.get_uniform_integer(0, len(remain_neg) - 1)
            # float dimension shrink
            if self.__dimension.get_type(temp_dim) == 0:
                if exa.get_feature(temp_dim) < neg_set[remain_neg[chosen_neg]].get_feature(temp_dim):
                    temp_v = self.__ro.get_uniform_double(exa.get_feature(temp_dim),
                                                          neg_set[remain_neg[chosen_neg]].get_feature(temp_dim))
                    if temp_v < self.__region[temp_dim][1]:
                        self.__region[temp_dim][1] = temp_v
                else:
                    temp_v = self.__ro.get_uniform_double(neg_set[remain_neg[chosen_neg]].get_feature(temp_dim),
                                                          exa.get_feature(temp_dim))
                    if temp_v > self.__region[temp_dim][0]:
                        self.__region[temp_dim][0] = temp_v
                r_i = 0
                while r_i < len(remain_neg):
                    if neg_set[remain_neg[r_i]].get_feature(temp_dim) < self.__region[temp_dim][0] or \
                            neg_set[remain_neg[r_i]].get_feature(temp_dim) > self.__region[temp_dim][1]:
                        remain_neg.remove(remain_neg[r_i])
                    else:
                        r_i += 1
            # integer dimension shrink
            elif self.__dimension.get_type(temp_dim) == 1:
                if exa.get_feature(temp_dim) < neg_set[remain_neg[chosen_neg]].get_feature(temp_dim):
                    temp_v = self.__ro.get_uniform_integer(exa.get_feature(temp_dim),
                                                           neg_set[remain_neg[chosen_neg]].get_feature(temp_dim) - 1)
                    if temp_v < self.__region[temp_dim][1]:
                        self.__region[temp_dim][1] = temp_v
                else:
                    temp_v = self.__ro.get_uniform_integer(neg_set[remain_neg[chosen_neg]].get_feature(temp_dim) - 1,
                                                           exa.get_feature(temp_dim))
                    if temp_v > self.__region[temp_dim][0]:
                        self.__region[temp_dim][0] = temp_v
                if self.__region[temp_dim][0] == self.__region[temp_dim][1]:
                    chosen_dim.append(temp_dim)
                    non_chosen_dim.remove(temp_dim)
                    self.__label[temp_dim] = False
                r_i = 0
                while r_i < len(remain_neg):
                    if neg_set[remain_neg[r_i]].get_feature(temp_dim) < self.__region[temp_dim][0] or \
                            neg_set[remain_neg[r_i]].get_feature(temp_dim) > self.__region[temp_dim][1]:
                        remain_neg.remove(remain_neg[r_i])
                    else:
                        r_i += 1
            # categorical
            else:
                chosen_dim.append(temp_dim)
                non_chosen_dim.remove(temp_dim)
                self.__label[temp_dim] = False
                r_i = 0
                while r_i < len(remain_neg):
                    if neg_set[remain_neg[r_i]].get_feature(temp_dim) != exa.get_feature(temp_dim):
                        remain_neg.remove(remain_neg[r_i])
                    else:
                        r_i += 1

        while len(non_chosen_dim) > self.__uncertain_bit:
            temp_dim = non_chosen_dim[self.__ro.get_uniform_integer(0, len(non_chosen_dim) - 1)]
            chosen_dim.append(temp_dim)
            non_chosen_dim.remove(temp_dim)
            self.__label[temp_dim] = False

        return dist_count

    def generate_inputs(self, ins, new_ins):
        trajectory = []
        if True:
            array_best = []
            for i in range(len(ins.get_features())):
                array_best.append(ins.get_features()[i])
            new_sam = []
            for i in range(len(new_ins.get_features())):
                new_sam.append(new_ins.get_features()[i])
            sorted_neg = sorted(self.__pop, key=lambda instance: instance.get_fitness())
            for i in range(self.__sample_size):
                trajectory.append((np.array(sorted_neg[i].get_features()) - np.array(array_best)).tolist())
            trajectory.append(new_sam)
            trajectory = [trajectory]
        return trajectory

    # inputs is 4D list
    def predict(self, inputs):

        # print 'in prediction-----'
        inputs = torch.from_numpy(np.array(inputs)).float()
        inputs = Variable(inputs.cuda())

        # print 'input: ', inputs
        # inputs = Variable(inputs)

        outputs = []
        for i in range(len(self.__predictors)):
            # print i, ' predictor---'
            output = self.__predictors[i].predictor(inputs)
            # print 'out 1: ', output
            output = output.data.cpu().numpy()
            # print 'out numpy: ', output
            output = output.reshape(output.size).tolist()
            # print 'out list: ', output
            outputs.append(output)
        # outputs = np.array(outputs)
        # print 'all out: ', outputs
        # outputs = np.mean(outputs, axis=0).tolist()
        # print 'mean out: ', outputs
        # print '-------------------------------'
        return outputs

    def delete_predictor(self, prob_matrix, truth_index, truth_label):

        label_threshold = 0.5
        del_index = []
        # delete all predictors which make mistakes
        if True:
            for i in range(len(prob_matrix)):
                if prob_matrix[i][truth_index] > label_threshold:
                    this_label = 1
                else:
                    this_label = 0
                if this_label != truth_label:
                    del_index.append(i)

        # delete the predictors which can't find the positive
        if False:
            if truth_label == 1:
                for i in range(len(prob_matrix)):
                    if prob_matrix[i][truth_index] > label_threshold:
                        this_label = 1
                    else:
                        this_label = 0
                    if this_label != truth_label:
                        del_index.append(i)

        # delete predictors
        new_predictors = []
        for i in range(len(self.__predictors)):
            if i not in del_index:
                new_predictors.append(self.__predictors[i])
        self.__predictors = new_predictors

        return

    # sequential Racos for mixed optimization
    # the dimension type includes float, integer and categorical
    def exp_ada_mix_opt(self, obj_fct=None, ss=2, bud=20, pn=1, rp=0.95, ub=1, at=5, step=1):

        sample_count = 0
        all_dist_count = 0
        flag = False
        res = []

        # initialize sample set
        self.clear()
        self.log_clear()
        self.set_parameters(ss=ss, bud=bud, pn=pn, rp=rp, ub=ub, at=at)
        self.reset_model()
        self.initialize(obj_fct)

        # ------------------------------------------------------
        # print 'after initialization------------'
        # self.show_pos_pop()
        # self.show_pop()
        # ------------------------------------------------------

        # optimization
        budget_c = self.__sample_size + self.__positive_num
        while budget_c < self.__budget:
            budget_c += 1
            res.append(self.__optimal.get_fitness())
            if budget_c % 10 == 0:
                # print '======================================================'
                print('budget ', budget_c, ':', self.__optimal.get_fitness())
                flag = True
                # self.__optimal.show_instance()
            adv_samples = []
            adv_inputs = []
            for adv_i in range(self.__adv_threshold):
                while True:
                    self.reset_model()
                    chosen_pos = self.__ro.get_uniform_integer(0, self.__positive_num - 1)
                    model_sample = self.__ro.get_uniform_double(0.0, 1.0)
                    if model_sample <= self.__rand_probability:
                        dc = self.shrink_model(self.__pos_pop[chosen_pos], self.__pop)
                        all_dist_count += dc

                    # -----------------------------------------------------------
                    # self.show_region()
                    # self.show_label()
                    # -----------------------------------------------------------

                    ins = self.pos_random_mix_isntance(self.__pos_pop[chosen_pos], self.__region, self.__label)

                    sample_count += 1

                    if (self.instance_in_list(ins, self.__pos_pop, self.__positive_num) is False) and (
                            self.instance_in_list(ins, self.__pop, self.__sample_size) is False):
                        # ins.set_fitness(obj_fct(ins.get_features()))
                        # ------------------------------------------------------
                        # print 'new sample:-------------------'
                        # ins.show_instance()
                        # print '------------------------------'
                        # ------------------------------------------------------

                        break
                this_input = self.generate_inputs(self.__pos_pop[chosen_pos], ins)
                adv_inputs.append(this_input)
                adv_samples.append(ins)

            # print 'inputs: ', len(adv_inputs), '*', len(adv_inputs[0]), '*', len(adv_inputs[0][0]), '*', len(adv_inputs[0][0][0])
            # get probability matrix, each line means a result list for a predictor
            probs, prob_matrix = self.__expert.predict(adv_inputs)
            # print '------------------------------------------'
            # print 'probs: ', probs
            max_index = probs.index(max(probs))
            # print 'max index: ', max_index
            good_sample = adv_samples[max_index]
            good_sample.set_fitness(obj_fct(good_sample.get_features()))

            if good_sample.get_fitness() < self.__optimal.get_fitness():
                truth_label = 1
            else:
                truth_label = 0

            t = np.array(prob_matrix)[:, max_index].T.tolist()
            self.__expert.update_weights(np.array(prob_matrix)[:, max_index].T, truth_label, flag)
            flag = False

            self.online_update(good_sample)

            # ------------------------------------------------------
            # print 'after updating------------'
            # self.show_pos_pop()
            # self.show_pop()
            # ------------------------------------------------------

            self.update_optimal()
        # print 'average sample times of each sample:', float(sample_count) / self.__budget
        # print 'average shrink times of each sample:', float(all_dist_count) / sample_count

        return res
