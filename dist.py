#!/usr/bin/python3
# coding: utf-8
'''
 @Time    : 2021/6/6 16:30
 @Author  : Shulu Chen
 @FileName: dist.py
 @Software: PyCharm
'''

import scipy.stats as st
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import math

# sns.distplot(random.normal(size=10000,scale=5,loc=50), hist=False,label="Class 1")
# sns.distplot(random.normal(size=10000,scale=5,loc=75), hist=False,label="Class 2")
# sns.distplot(random.normal(size=10000,scale=5,loc=100), hist=False,label="Class 3")
#
# sns.distplot(random.normal(size=10000,scale=math.sqrt(50),loc=125), hist=False,label="Class 1,2")
# sns.distplot(random.normal(size=10000,scale=math.sqrt(75),loc=225), hist=False,label="Class 1,3")
#
# plt.legend()
# plt.show()

print(125+st.norm.ppf(1-200/340)*7.07)
# print(st.norm.cdf(53.4,loc=50,scale=5))