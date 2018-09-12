# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:02:53 2018

@author: Victor Costa

Module with benchmark functions to test continuous optimization methods.
Function expressions extracted from https://www.sfu.ca/~ssurjano/optimization.html
"""

from math import pi
import numpy as np


def sphere(x):
    """ Reference: https://www.sfu.ca/~ssurjano/spheref.html
        Number of dimensions: arbitrary.
        Input domain: -5.12 <= xi <= 5.12 (i = 1,..., d).
        Global minimum: f(x*) = 0 at x* = (0,..., 0). """
       
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return sum(x*np.transpose(x))

def rosenbrock(x):
    """ Reference: https://www.sfu.ca/~ssurjano/rosen.html
        Number of dimensions: arbitrary.
        Input domain: -5 <= xi <= 10 (i = 1,..., d).
        Global minimum: f(x*) = 0 at x* = (1,..., 1). """
        
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def ackley(x, a=20, b=0.2, c=2*pi):
    """ Reference: https://www.sfu.ca/~ssurjano/ackley.html
        Number of dimensions: arbitrary.
        Input domain: -32.768 <= xi <= 32.768 (i = 1,..., d).
        Global minimum: f(x*) = 0 at x* = (0,..., 0). """
        
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( np.cos( c * x ))
    return -a * np.exp(-b*np.sqrt( s1 / n )) - np.exp( s2 / n ) + a + np.exp(1)