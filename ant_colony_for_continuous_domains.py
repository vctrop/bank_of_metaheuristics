# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:36:48 2018

@author: Victor O. Costa

Implementation of the Ant Colony for Continuous Domains method from Socha, 2008.
"""
    
import numpy as np
from scipy.stats import norm

class ACOr:
    """ Class for the Ant Colony Optimization for Continuous Domains, following (Socha and Dorigo, 2006) """

    def __init__(self):
        """ Constructor """
        self.verbosity = True
        
        # Initial algorithm parameters
        self.num_iter = 0                               # Number of iterations
        self.pop_size = 5                               # Population size
        self.k = 50                                     # Archive size
        self.q = 0.1                                    # Locality of search
        self.xi = 0.85                                  # Speed of convergence
        
        # Initial (NULL) problem definition
        self.num_var = None                             # Number of variables
        self.initial_ranges = []                        # Initialization boundaries for each variable
        self.is_constrained = []                        # Here, if a variable is constrained, it will be limited to its initialization boundaries for all the search
        self.cost_function = None                       # Cost function to guide the search
        
        # Optimization results
        self.SA = None                                  # Solution Archive
        self.best_solution = None                       # Best solution of the archive

        
    def set_verbosity(self, status):
        """ If verbosity is set True, print partial results of the search will be printed """
        if not (type(status) is bool):
            print("Error, verbosity parameter must be a boolean")
            exit(-1)
        self.verbosity = status
    
        
    def set_parameters(self, num_iter, pop_size, k, q, xi):
        """ Define values for the parameters used by the algorithm """
        if num_iter <= 0:
            print("Number of iterations must be greater than zero")
            exit(-1)
        self.num_iter = num_iter
        self.pop_size = pop_size
        self.k = k
        self.q = q
        self.xi = xi

    
    def define_variables(self, initial_ranges, is_bounded):
        """ Defines the number of variables, their initial values ranges and wether or not these ranges constrain the variable during the search """
        # Error checking
        if self.num_iter == 0:
            print("Error, please set algorithm parameters before variables definition")
            exit(-1)
        if len(initial_ranges) == 0 or len(is_bounded) == 0:
            print("Error, ranges and is_bounded lists must not be empty")
            exit(-1)
        
        self.num_var = len(initial_ranges)
        self.initial_ranges = initial_ranges
        self.is_bounded = is_bounded
        self.SA = np.zeros((self.k, self.num_var + 1))
        
        
            
    def set_cost(self, costf):
        """ Sets the cost function that will guide the search """
        self.cost_function = costf
        
    
    def _biased_selection(self, probabilities):
        """ Returns an index based on a set of probabilities (also known as roulette wheel selection in GA) """
        r = np.random.uniform(0, sum(probabilities))
        for i, f in enumerate(probabilities):
            r -= f
            if r <= 0:
                return i
         
         
    def optimize(self):
        """ Initializes the archive and enter the main loop, until it reaches maximum number of iterations """
        # Sanity check
        if self.num_var == 0:
            print("Error, first set the number of variables and their boundaries")
            exit(-1)
        
        if self.cost_function == None:
            print("Error, first define the cost function to be used")
            exit(-1)
            
        if self.verbosity:   print("[INITIALIZING SOLUTION ARCHIVE]")
        # Initialize the archive by random sampling, respecting each variable's constraints
        pop = np.zeros((self.pop_size, self.num_var +1))
        w = np.zeros(self.k)
        
        for i in range(self.k):
            for j in range(self.num_var): 
                self.SA[i, j] = np.random.uniform(self.initial_ranges[j][0], self.initial_ranges[j][1])        # Initialize solution archive randomly
            self.SA[i, -1] = self.cost_function(self.SA[i, 0:self.num_var])                            # Get initial cost for each solution
        self.SA = self.SA[self.SA[:, -1].argsort()]                                                    # Sort solution archive (best solutions first)

        x = np.linspace(1,self.k,self.k) 
        w = norm.pdf(x,1,self.q*self.k)                                 # Weights as a gaussian function of rank with mean 1, std qk
        p = w/sum(w)                                                    # Probabilities of selecting solutions as search guides
        
        if self.verbosity:   print("ALGORITHM MAIN LOOP")
        
        # Algorithm runs until it reaches the determined number of iterations
        for iteration in range(self.num_iter):
            if self.verbosity:
                print("[%d]" % iteration)
                # print(self.SA[0, :])
            
            Mi = self.SA[:, 0:self.num_var]                                                                     # Matrix of means
            for ant in range(self.pop_size):                                                                   # For each ant in the population
                l = self._biased_selection(p)                                                                   # Select solution of the SA to sample from based on probabilities p
                
                for var in range(self.num_var):                                                                # Calculate the standard deviation of all variables from solution l
                    sigma_sum = 0
                    for i in range(self.k):
                        sigma_sum += abs(self.SA[i, var] - self.SA[l, var])
                    sigma = self.xi * (sigma_sum/(self.k - 1))
                     
                    pop[ant, var] = np.random.normal(Mi[l, var], sigma)                                         # Sample from normal distribution with mean Mi and st. dev. sigma
                    
                    # Search space boundaries violation is only dealt with when the variable is considered bounded (self.is_bounded)
                    if self.is_bounded[var]:
                        # Use the hard border strategy
                        if pop[ant, var] < self.initial_ranges[var][0]:
                            pop[ant, var] = self.initial_ranges[var][0]
                        elif pop[ant, var] > self.initial_ranges[var][1]:
                            pop[ant, var] = self.initial_ranges[var][1]        
                        
                        # Use the random position strategy
                        # if pop[ant, var] < self.initial_ranges[var][0] or pop[ant, var] > self.initial_ranges[var][1]:                   
                            # pop[ant, var] = np.random.uniform(self.initial_ranges[var][0], self.initial_ranges[var][1])
                    
                    
                pop[ant, -1] = self.cost_function(pop[ant, 0:self.num_var])                                     # Evaluate cost of new solution
                
            self.SA = np.append(self.SA, pop, axis = 0)                                                         # Append new solutions to the Archive
            self.SA = self.SA[self.SA[:, -1].argsort()]                                                         # Sort solution archive according to the fitness of each solution
            print(self.SA)
            self.SA = self.SA[0:self.k, :]                                                                      # Remove worst solutions
        
        self.best_solution = self.SA[0, :]
        return self.best_solution  

# end class 