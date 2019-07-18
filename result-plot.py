import numpy as np
import matplotlib.pyplot as plt
path = '/home/amax/Desktop/ExpAdaptation'

a=[]
for j in range(5):
 f = open(path + '/Results/SyntheticProbs/sphere/dimension10/opt-log-sphere-dim10-bias0.5.txt')
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
    a.append((float(line.split(': ')[1].split('#')[0]),dist))

# print(min(errors))
# plt.scatter(dists,errors)
# plt.show()

a.sort(key=lambda a:a[1])
errors=[x[0] for x in a]
dists=[x[1] for x in a]

s=0
t=0
stride=0.1
y=[]
x=[stride]
while t<2000:
    while dists[t]<=x[-1]:
        t+=1
        if t==2000:break
    if s!=t:
        y.append(np.mean(errors[s:t]))
        s=t
        x.append(x[-1]+stride)
    else:
        x[-1]+=stride

del x[-1]
plt.scatter(x,y)
plt.show()


