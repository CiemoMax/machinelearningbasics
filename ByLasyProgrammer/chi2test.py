# -*- coding: utf-8 -*-
# @Time    : @Date
import logging
import pandas as pd
import os
import numpy as np


## this is chi2 test from Udamy course: Bayesian Machine Learning in Python: A/B Testing


class DataGenerator:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def next(self):
        click1 = 1 if (np.random.random() > self.p1) else 0
        click2 = 1 if (np.random.random() > self.p2) else 0
        return click1, click2


def get_p_value(T):
    # 				negative	positive	sub_total
    # situation A:		a			b			a+b
    # situation B:		c			d			c+d
    #	sub_total:		a+c			b+d			a+b+c+d
    #
    # chi2_value = (a*d - b*c)^2 * (a+b+c+d)/[(a+b)*(c+d)*(a+c)*(b+d)]

    det = float(T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0])
    chi2 = det / T[0].sum() * det / T[1].sum() * T.sum() / T[:, 0].sum() / T[:, 1].sum()
    p = 1 - chi2.cdf(x=chi2, df=1)
    return p


def run_experiment(p1, p2, N):
    data = DataGenerator(p1, p2)
    p_values = np.empty(N)
    T = np.zeros((2, 2)).astype(np.float32)

    for i in range(N):
        c1, c2 = data.next()
    T[0, c1] += 1
    T[1, c2] += 1

    if i < 10:
        p_values[i] = None
    else:
        p_values[i] = get_p_value(T)

    plt.plot(p_values)
    plt.plot(np.ones(N) * 0.05)
    plt.show()


run_experiment(0.1, 0.11, 20000)