# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 18:05:07 2017

@author: nbetar
"""
import numpy as np
import numpy.linalg as linalg

class PriceModel(object):
    """A generic price model based on a finite set of prices and a transition
    matrix describing the transition probability between two states.

    Attributes:

    price set
    transition_matrix
    """
    def __init__(self,price_set,transition_matrix):
        self.price_set = price_set
        self.transition_matrix = transition_matrix

    def compute_output_state(self,input_state,n_iter):
        output_state = np.dot(linalg.matrix_power(self.transition_matrix,n_iter),input_state)
        return output_state

class BinomialPriceModel(PriceModel):
    """A binomial price model with recombining tree.

    Attributes:

    a price set
    """

    def __init__(self,price_set):
        self.price_set = price_set

        m = len(price_set)/2

        N = 2*m+1

        A = 0.5*(np.diag(np.ones(N-1)) + np.diag(np.ones(N-3),2))

        B = np.zeros((1,N-1))
        B[0,0]=0.5
        B[0,1]=0.5

        A=np.concatenate((B,A),axis=0)

        B = np.zeros((N,1))
        B[-1,0]=0.5
        B[-2,0]=0.5

        self.transition_matrix=np.concatenate((A,B),axis=1)


class TrinomialPriceModel(PriceModel):
    """A trinomial price model with recombining tree.

    Attributes:
    """

    def __init__(self,price_set,up,down):
        self.price_set = price_set
        self.up = up
        self.down = down
        self.flat = 1 - (up+down)

        m = len(price_set)/2

        N = 2*m+1

        A = self.flat * np.diag(np.ones(N)) + up * np.diag(np.ones(N-1),-1) + down * np.diag(np.ones(N-1),1)
        A[0,0] = A[0,0] + down
        A[N-1,N-1] = A[N-1,N-1] + up

        self.transition_matrix=A
