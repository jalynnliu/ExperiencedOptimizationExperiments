'''
Demos for RACOS

Author:
    Yi-Qi Hu

time:
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

from Racos import RacosOptimization
from Components import Dimension
from ObjectiveFunction import Sphere
from ObjectiveFunction import Ackley
from ObjectiveFunction import SetCover
from ObjectiveFunction import DistributedFunction
from ObjectiveFunction import three_types_function as tt_func
# from TrainPolicy import TrainingPolicy
from Tools import list2string
from Tools import RandomOperator
# from keras.models import model_from_json
import numpy as np
import xlwt
import time

# parameters
SampleSize = 20             # the instance number of sampling in an iteration
MaxIteration = 100          # the number of iterations
Budget = 200                # budget in online style
PositiveNum = 2             # the set size of PosPop
RandProbability = 0.99      # the probability of sample in model
UncertainBits = 2           # the dimension size that is sampled randomly


def result_analysis(res, top):
    res.sort()
    top_k = []
    for i in range(top):
        top_k.append(res[i])
    mean_r = np.mean(top_k)
    std_r = np.std(top_k)
    print(mean_r, '#', std_r)
    return mean_r, std_r


def time_formulate(start_t, end_t):
    time_l = end_t - start_t
    if time_l < 0:
        print('time error!')
        hour = 0
        minute = 0
        second = 0
    else:
        hour = int(time_l / 3600)
        time_l = time_l - hour*3600
        minute = int(time_l / 60)
        second = time_l - minute*60
    return hour, minute, second


def xls_write(xls_buff, path='ExpRacos_test_log.xls'):

    xls_index = 0

    workbook = xlwt.Workbook()
    data_sheet = workbook.add_sheet('sheet1', cell_overwrite_ok=True)

    for i in range(len(xls_buff)):
        if i == 0:
            data_sheet.write(xls_index, 0, xls_buff[i])
        else:
            func_buff = xls_buff[i]
            for j in range(len(func_buff[0])):
                for k in range(len(func_buff)):
                    data_sheet.write(xls_index, k, func_buff[k][j])
                xls_index += 1
        xls_index += 2

    workbook.save(path)
    print('ExpRacos testing log saved!')
    return


def run_mix_racos():

    # parameters
    sample_size = 8            # the instance number of sampling in an iteration
    budget = 20000                # budget in online style
    positive_num = 2            # the set size of PosPop
    rand_probability = 0.99     # the probability of sample in model
    uncertain_bit = 2           # the dimension size that is sampled randomly

    repeat = 4
    list_budget = [100, 1000, 10000, 50000]

    # dimension setting
    dimension_size = 15
    float_region = [-100, 100]
    integer_region = [-100, 100]
    categorical_region = [0, 2]

    dimension = Dimension()
    dimension.set_dimension_size(dimension_size)
    for i in range(dimension_size):
        if i % 3 == 0:
            dimension.set_region(i, float_region, 0)
        elif i % 3 == 1:
            dimension.set_region(i, integer_region, 1)
        else:
            dimension.set_region(i, categorical_region, 2)


    # optimization
    racos = RacosOptimization(dimension)

    for i in range(repeat):

        start_t = time.time()
        racos.mix_opt(tt_func, ss=sample_size, bud=list_budget[i], pn=positive_num, rp=rand_probability, ub=uncertain_bit)
        end_t = time.time()

        optimal = racos.get_optimal()

        hour, minute, second = time_formulate(start_t, end_t)

        print('total budget is ', list_budget[i], '------------------------------')
        print('spending time: ', hour, ' hours ', minute, ' minutes ', second, ' seconds')
        print('optimal value: ', optimal.get_fitness())


if __name__ == '__main__':
    run_mix_racos()
