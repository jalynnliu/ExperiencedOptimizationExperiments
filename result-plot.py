import numpy as np
import matplotlib.pyplot as plt

path = '/data/ExpAdaptation'


def meanScatter():
    a = []
    problem_name = 'sphere'
    for j in range(1):
        f = open(
            path + '/Results/SyntheticProbs/' + problem_name + '/dimension10/opt-log-' + problem_name + '-dim10-bias0.5.txt')
        for i in range(2000):
            line = f.readline()
            while line.split(' ')[0] != 'source':
                line = f.readline()
            line = line.split(': ')[1]
            biases = line.split(' ')
            dist = 0
            for bias in biases:
                dist += abs(float(bias) - 0.1)
            line = f.readline()
            while line.split(': ')[0] != 'optimization result':
                line = f.readline()
            a.append((float(line.split(': ')[1].split('#')[0]), dist))

    a.sort(key=lambda a: a[1])
    errors = [x[0] for x in a]
    dists = [x[1] for x in a]

    s = 0
    t = 0
    stride = 0.1
    y = []
    x = [stride]
    while t < len(a):
        while dists[t] <= x[-1]:
            t += 1
            if t == len(a): break
        if s != t:
            y.append(np.mean(errors[s:t]))
            s = t
            x.append(x[-1] + stride)
        else:
            x[-1] += stride

    del x[-1]
    plt.scatter(x, y)
    plt.show()


def twoColor():
    f = open(path + '/Results/SyntheticProbs/ExperimentTwo/opt-log-sphere-dim10-bias0.5remix.txt')
    c = ['red', 'blue']
    for j in range(2):
        a = []
        for i in range(1000):
            line = f.readline()
            while line.split(' ')[0] != 'source':
                line = f.readline()
            line = line.split(': ')[1]
            biases = line.split(' ')
            dist = 0
            for bias in biases:
                dist += abs(float(bias) - 0.1)
            line = f.readline()
            while line.split(': ')[0] != 'optimization result':
                line = f.readline()
            a.append((float(line.split(': ')[1].split('#')[0]), dist))

        x = [x[0] for x in a]
        y = [x[1] for x in a]

        plt.scatter(x, y, c=c[j])
    plt.show()


meanScatter()
