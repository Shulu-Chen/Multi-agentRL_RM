#!/usr/bin/python3
# coding: utf-8
'''
 @Time    : 2021/5/25 23:05
 @Author  : Shulu Chen
 @FileName: PaxBehavior.py
 @Software: PyCharm
'''
import math
import matplotlib.pyplot as plt
from numpy import log as ln
from scipy.stats import norm
from numpy import random
import numpy as np
from scipy.stats import beta
emult_l=1.5
basefare_l=100*2
emult_h=2
basefare_h=250*2
alpha_l = 2
beta_l = 2.5
alpha_h=10
beta_h=4.6
fClassMeanArrivals_l=50
fClassMeanArrivals_h=20
bookingHorizon = 50
nFareClasses=2
nEpisode=1000
##Exp distribution

# x=[]
# for t in range(182):
#     x.append(0.6+1.2/182*t)
# plt.plot(x)
# plt.show()

# day=50
#
# for day in range(181):
# basefare_l=200+day*1.5
# basefare_h=500+day*1.5
# emult_l=1.5+day/30
# emult_h=2+day/30
# prob_h=[]
# prob_l=[]
# for f in range(800):
#     prob_l.append(min(1,math.exp((-ln(2)*(f-basefare_l))/((emult_l-1)*basefare_l))))
#     prob_h.append(min(1,math.exp((-ln(2)*(f-basefare_h))/((emult_h-1)*basefare_h))))
# plt.plot(prob_h,label="high")
# plt.plot(prob_l,label="low")
# plt.vlines(100, 0, 1,colors="r")
# plt.vlines(275,0,1,colors="r")
# plt.vlines(250, 0, 1,colors="g")
# plt.vlines(425,0,1,colors="g")
# plt.title("WTP day="+str(day))
# plt.legend()
# #
# # plt.savefig('./graph/'+str(day)+".png")
# plt.show()
# day=182
#
# basefare_l=100+day/2
# basefare_h=250+day/2
# emult_l=1.5+day/100
# emult_h=2+day/100
# print(basefare_h,basefare_l,emult_h,emult_l)
# print(min(1,math.exp((-ln(2)*(125-basefare_l))/((emult_l-1)*basefare_l))))
# print(min(1,math.exp((-ln(2)*(250-basefare_l))/((emult_l-1)*basefare_l))))
# print(min(1,math.exp((-ln(2)*(275-basefare_h))/((emult_h-1)*basefare_h))))
# print(min(1,math.exp((-ln(2)*(400-basefare_h))/((emult_h-1)*basefare_h))))
x1=np.arange(1,51)/50
x2=np.arange(0,50)/50
lambdaPrimeValues_l = beta.cdf(x1, alpha_l, beta_l)-beta.cdf(x2, alpha_l, beta_l)
lambdaPrimeValues_h = beta.cdf(x1, alpha_h, beta_h)-beta.cdf(x2, alpha_h, beta_h)
lambdaValues_l = lambdaPrimeValues_l*fClassMeanArrivals_l
lambdaValues_h = lambdaPrimeValues_h*fClassMeanArrivals_h
nArrivals = np.zeros((nFareClasses,bookingHorizon,nEpisode))
for i in range(5000):
    for step_t in range(bookingHorizon):
        nArrivals[0][step_t,:] = np.random.poisson(lam=lambdaValues_l[step_t],size=nEpisode)
        nArrivals[1][step_t,:]= np.random.poisson(lam=lambdaValues_h[step_t],size=nEpisode)

a=[sum(nArrivals[0][i,:])/nEpisode for i in range(49)]
b=[sum(nArrivals[1][i,:])/nEpisode for i in range(49)]
plt.plot(a,label="class L")
plt.plot(b,label="class_H")
plt.title("pax arrival dist.")
plt.legend()
plt.show()

def generate_pax():
    x1=np.arange(1,51)/50
    x2=np.arange(0,50)/50
    lambdaPrimeValues_l = beta.cdf(x1, alpha_l, beta_l)-beta.cdf(x2, alpha_l, beta_l)
    lambdaPrimeValues_h = beta.cdf(x1, alpha_h, beta_h)-beta.cdf(x2, alpha_h, beta_h)
    lambdaValues_l = lambdaPrimeValues_l*fClassMeanArrivals_l
    lambdaValues_h = lambdaPrimeValues_h*fClassMeanArrivals_h
    nArrivals = np.zeros((nFareClasses,bookingHorizon))
    for step_t in range(bookingHorizon):
        nArrivals[0][step_t] = np.random.poisson(lam=lambdaValues_h[step_t])
        nArrivals[1][step_t]= np.random.poisson(lam=lambdaValues_l[step_t])
    # print(nArrivals)
    # nArrivals=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0],#10
    #            [0, 0, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 0, 0, 2, 0, 0]]#20
    return nArrivals
# test_data=generate_pax()

def get_pax(data,day):
    return [int(data[0][day]),int(data[1][day])]

def PaxProb(price,class_level,day):
    ad_basefare_h=basefare_h+day*1.5
    ad_basefare_l=basefare_l+day*1.5
    ad_emult_h=emult_h+day/30
    ad_emult_l=emult_l+day/30
    if class_level==0: #high_level
        prob=min(1,math.exp((-ln(2)*(price-ad_basefare_h))/((ad_emult_h-1)*ad_basefare_h)))
        choose=random.uniform(0,1,1)
        if choose[0]<=prob:
            return True
        else:
            return False
    elif class_level==1: #low_level
        prob=min(1,math.exp((-ln(2)*(price-ad_basefare_l))/((ad_emult_l-1)*ad_basefare_l)))
        choose=random.uniform(0,1,1)
        if choose[0]<=prob:
            return True
        else:
            return False


def Settlement(price, total_pax,class_level,day):
    r=0
    b=0
    for i in range(total_pax):
        # buy=PaxProb(price,class_level,day)
        buy=True
        if buy:
            r+=price
            b+=1
    return [r,b]

