# -*- coding: utf-8 -*-
# @Time    : @Date

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

num_trials = 2000
bandit_probabilities = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1

    def pull(self):
        return np.random.random() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        self.a += x
        self.b += 1-x

def plot(bandits, trial):
    x = np.linspace(0,1,200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label = 'real p: %f' % b.p)
    plt.title('Bandit distributions after %s trials' % trial)
    plt.legend()
    plt.show()

def experiment():
    bandits = [Bandit(p) for p in bandit_probabilities]
    sample_points = [5,10,20,50,100,200,500,1000,1500,1999]

    for i in range(num_trials):
        bestb = None
        maxsample = -1
        allsamples = []

        for b in bandits:
            sample = b.sample()
            allsamples.append('%f' % sample)
            if sample > maxsample:
                maxsample = sample
                bestb = b
            if i in sample_points:
                print ('current samples: %s' % allsamples)
                plot(bandits, i)

            x = bestb.pull()
            bestb.update(x)

if __name__ == '__main__':
    experiment()

