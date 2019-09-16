from __future__ import division, print_function
from smac.facade.func_facade import fmin_smac
from ObjectiveFunction import DistributedFunction
from Components import Dimension
from Tools import RandomOperator
import numpy as np

dimension_size = 10

dimension = Dimension()
dimension.set_dimension_size(dimension_size)
dimension.set_regions([[-0.5, 0.5] for _ in range(dimension_size)], [0 for _ in range(dimension_size)])

func = DistributedFunction(dimension, bias_region=[-0.5, 0.5])
target_bias = [0.25 for _ in range(dimension_size)]
func.setBias(target_bias)

ro = RandomOperator()
prob_fct = func.DisRosenbrock
x0 = [ro.get_uniform_double(-0.5, 0.5) for _ in range(dimension_size)]
ans = []
for i in range(10):
    x, cost, _ = fmin_smac(func=prob_fct, x0=x0, bounds=[[-0.5, 0.5] for _ in range(dimension_size)], maxfun=50, rng=3)
    ans.append(x)
# print("Optimum at {} with cost of {}".format(x, cost))
print(np.mean(ans))
print(np.std(ans))
