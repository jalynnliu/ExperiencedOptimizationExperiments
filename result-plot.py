import numpy as np
import matplotlib.pyplot as plt

f=open('Results/SyntheticProbs/sphere/dimension10/opt-log-sphere-dim10-bias0.5.txt')
errors=[]
dists=[]
for i in range(2000):
    line=f.readline()
    while line.split(' ')[0]!='source':
        line=f.readline()
    line=line.split(': ')[1]
    biases=line.split(' ')
    dist=0
    for bias in biases:
        dist+=abs(float(bias)-0.1)
    line = f.readline()
    while line.split(': ')[0] != 'optimization result':
        line = f.readline()
    errors.append(float(line.split(': ')[1].split('#')[0]))
    dists.append(dist)

print(min(errors))
plt.scatter(dists,errors)
plt.show()
