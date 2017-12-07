# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 18:03:13 2017

@author: nbetar
"""
import numpy as np
#import numpy.linalg as linalg
import time
#from PriceModel import PriceModel
#from PriceModel import BinomialPriceModel
from PriceModel import TrinomialPriceModel

m = 10

N = 2*m+1

p = range(-m,m+1)

up = 0.2
down = 0.3

tpm = TrinomialPriceModel(p,up,down)
#bpm = BinomialPriceModel(p)

input_state = np.zeros((N,1))
input_state[m,0] = 1 

#input_state = np.mat(np.ones((N,1)))

t = time.time()

output_state = tpm.compute_output_state(input_state,100)

elapsed = time.time() - t