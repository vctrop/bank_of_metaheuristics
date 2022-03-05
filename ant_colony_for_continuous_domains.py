#!python3

# Copyright (C) 2020  Victor O. Costa

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
# Python standar lib
import math
# 3rth party
import numpy as np
# Own
from base_metaheuristic import Base

class ACOr(Base):
    """ Class for the Ant Colony Optimization for Continuous Domains, following (Socha and Dorigo, 2006) """

    def __init__(self):
        """ Constructor """
        # Define verbosity and NULL problem definition
        super().__init__()
        
        # Initial algorithm parameters
        self.relative_iterations = None                 # Array containing the iterations at which best solutions are reported
        self.num_iter = 0                               # Number of iterations
        self.pop_size = 5                               # Population size
        self.k = 50                                     # Archive size
        self.q = 0.01                                   # Locality of search (selection of pivot ants)
        self.xi = 0.85                                  # Speed of convergence (spreadness of ant generation)
        
        # Optimization results
        self.SA = None                                  # Solution Archive
        self.best_solution = None                       # Best solution of the archive
        

    def set_parameters(self, pop_size, k, q, xi, function_evaluations_array):
        """ Define values for the parameters used by the algorithm """
        # Input error checking
        if len(function_evaluations_array) == 0:
            print("Error, objective function evaluation array must not be empty")
            exit(-1)
        if pop_size <= 0 or k <= 0 or q <= 0 or xi <= 0:
            print("Error, parameters must be non-null positives")
            exit(-1)
            
        
        # Number of function evaluations for ACOr: pop_size * num_iterations
        function_evaluations_array = np.array(function_evaluations_array)
        self.relative_iterations = (function_evaluations_array - k) / pop_size
        all_divisible = (np.array([x.is_integer() for x in self.relative_iterations])).all()
        if not all_divisible:
            print("Error, at least one number of function evaluations subtracted by k is not divisible by population size m")
            exit(-1)
        
        self.num_iter = int(np.max(self.relative_iterations))
        self.pop_size = pop_size
        self.k = k
        self.q = q
        self.xi = xi

    
    def define_variables(self, initial_ranges, is_bounded):
        """ Defines the number of variables, their initial values ranges and wether or not these ranges constrain the variable during the search """
        # Input error checking
        if self.num_iter == 0:
            print("Error, trying to define variables before setting algorithm parameters or using k = num_iter")
            exit(-1)
        if len(initial_ranges) == 0 or len(is_bounded) == 0:
            print("Error, initial_ranges and is_bounded lists must not be empty")
            exit(-1)
        if len(initial_ranges) != len(is_bounded):
            print("Error, the number of variables for initial_ranges and is_bounded must be equal")
            exit(-1)
        
        self.num_variables = len(initial_ranges)
        self.initial_ranges = initial_ranges
        self.is_bounded = is_bounded
        self.SA = np.zeros((self.k, self.num_variables + 1))

    
    def _biased_selection(self, probabilities):
        """ Returns an index based on a set of probabilities (also known as roulette wheel selection in GA) """
        r = np.random.uniform(0, sum(probabilities))
        for i, f in enumerate(probabilities):
            r -= f
            if r <= 0:
                return i
    

    def update_success_rate(self, success_count):
        """ Success rate is not updated in vanilla ACOr """
        pass
    
    def control_xi(self):
        """ Xi is not updated in vanilla ACOr """
        pass
    
    def control_q(self):
        """ q is not updated in vanilla ACOr """
        pass
    
    def gaussian_pdf_weights(self, x):
        gaus_std = self.q * self.k
        gaus_avg = 1
        w = (1 / (gaus_std * math.sqrt(2 * math.pi))) * np.exp( (-1/2) * ( ( (x - gaus_avg) / gaus_std ) ** 2) )
        
        return w
    
    def handle_adaptions(self, success_count):
        self.update_success_rate(success_count)
        self.control_q()
        self.control_xi()
    
    def optimize(self):
        """ Initializes the archive and enter the main loop, until it reaches maximum number of iterations """
        # Error checking
        if self.num_variables == None:
            print("Error, number of variables and their boundaries must be defined prior to optimization")
            exit(-1)
        if self.cost_function == None:
            print("Error, cost function must be defined prior to optimization")
            exit(-1)
        
        # Keep solutions defined by function_evaluations_array
        recorded_solutions = []
        
        # Initialize the archive by random sampling, respecting each variable's boundaries   
        if self.verbosity:   print("[INITIALIZING SOLUTION ARCHIVE]")
        pop = np.zeros((self.pop_size, self.num_variables +1))
        w = np.zeros(self.k)
        
        for i in range(self.k):
            for j in range(self.num_variables): 
                self.SA[i, j] = np.random.uniform(self.initial_ranges[j][0], self.initial_ranges[j][1])     # Initialize solution archive randomly
            self.SA[i, -1] = self.cost_function(self.SA[i, 0:self.num_variables])                           # Get initial cost for each solution
        self.SA = self.SA[self.SA[:, -1].argsort()]                                                         # Sort solution archive (best solutions first)
        
        # Array containing indices of solution archive position
        x = np.linspace(1,self.k,self.k) 
        w = self.gaussian_pdf_weights(x)                                         # Weights as a Gaussian function of rank with mean 1, std qk
        p = w/sum(w) 
        
        if self.verbosity:   print("ALGORITHM MAIN LOOP")
        # Algorithm runs until it reaches the determined number of iterations
        for iteration in range(self.num_iter):
            if self.verbosity:
                print("[%d]" % iteration)
                print(self.SA[0, :])
            
            success_count = 0                                                   # Count how many ant improve the solution they are sampling from    
            Mi = self.SA[:, 0:self.num_variables]                               # Matrix of means
            for ant in range(self.pop_size):                                    # For each ant in the population
                l = self._biased_selection(p)                                   # Select solution of the SA to sample from based on probabilities p
                # Compute average distances from the chosen solution to other solutions
                # Used as standard deviation of solution generation
                sigmas_array = self.xi * np.sum(np.abs(self.SA[:,:-1] - self.SA[l, :-1]), axis = 0) / (self.k - 1)
                
                for var in range(self.num_variables):
                    sigma = sigmas_array[var]
                    pop[ant, var] = np.random.normal(Mi[l, var], sigma)         # Sample from normal distribution with mean Mi and st. dev. sigma
                    
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
                    
                # Evaluate cost of new solution
                pop[ant, -1] = self.cost_function(pop[ant, 0:self.num_variables])       
                
                # Check if the new solution is better than the one the ant sampled from
                if pop[ant, -1] < self.SA[l, -1]:
                    success_count += 1
                    
            # Compute success rate, updates xi and q (No effect in vanilla ACOr)
            self.handle_adaptions(success_count)
            
            # Append new solutions to the Archive
            self.SA = np.append(self.SA, pop, axis = 0)                                                         
            # Update PDF from which ants sample their centers, according to updates in q parameter
            w = self.gaussian_pdf_weights(x)                                         # Weights as a gaussian function of rank with mean 1, std qk
            p = w/sum(w)                                                            # Probabilities of selecting solutions as search guides
        
            # Sort solution archive according to the fitness of each solution
            self.SA = self.SA[self.SA[:, -1].argsort()]                                                         
            # Remove worst solutions
            self.SA = self.SA[0:self.k, :]   
            # Extract current best solution
            self.best_solution = np.array(self.SA[0, :])
            if (self.relative_iterations - 1 == iteration).any():
                recorded_solutions.append(np.array(self.best_solution))
            
        return np.array(recorded_solutions)
        

# Success rate adaptive ACOr 
class SRAACOr(ACOr):
    """ Parent class of all adaptive versions of ACOr."""
    
    def __init__(self):
        """ Constructor """
        super().__init__()
        self.success_rate = None
        self.min =      {'q' : None,
                         'xi': None}
        self.max =      {'q' : None,
                         'xi': None}
        self.map_type = {'q' : None,
                         'xi': None}
                         
        self.lin_a =    {'q' : None,
                        'xi': None}
        self.lin_b =    {'q' : None,
                        'xi': None}
                        
        self.sig_K = 2
        self.sig_Q =    {'q' : None,
                        'xi': None}
        self.sig_B =    {'q' : None,
                         'xi': None}
                         
        self.exp_A =    {'q' : None,
                         'xi': None}
        self.exp_B =    {'q' : None,
                         'xi': None}
        
        
    def update_success_rate(self, success_count):
        """ Returns the success rate of the swarm at a given iteration,
            considering how many ants generated better solutions than the solutions they sampled from """
        self.success_rate = success_count / self.pop_size
       
       
    def parameterize_map(self, parameter):
        if not isinstance(parameter, str) or (parameter != 'q' and parameter != 'xi'):
            print('Parameter must be a string equal to \'q\' or \'xi\'')
            exit(-1)
        
        if self.map_type[parameter] == 'lin':
            self.lin_a[parameter] = self.max[parameter] - self.min[parameter]
            self.lin_b[parameter] = self.min[parameter]
        elif self.map_type[parameter] == 'sig':
            self.sig_Q[parameter] = (self.sig_K - self.min[parameter]) / self.min[parameter]
            self.sig_B[parameter] = math.log( (self.max[parameter] / (self.sig_K - self.max[parameter])) * self.sig_Q[parameter])
        else:
            self.exp_A[parameter] = self.min[parameter]
            self.exp_B[parameter] = math.log( self.max[parameter] / self.min[parameter] )
        
        
    def evaluate_map(self, parameter, x):
        if not isinstance(parameter, str) or (parameter != 'q' and parameter != 'xi'):
            print('Parameter must be a string equal to \'q\' or \'xi\'')
            exit(-1)
        
        if self.map_type[parameter] == None:
            print('Please first define the map type of ' + parameter)
            exit(-1)
        
        # Linear map
        if self.map_type[parameter] == 'lin':
            if self.lin_a[parameter] == None or self.lin_b[parameter] == None:
                print('Error, first parameterize the line')
                exit(-1)
            y = self.lin_a[parameter] * x + self.lin_b[parameter]
        # Sigmoidal map
        elif self.map_type[parameter] == 'sig':
            if self.sig_Q[parameter] == None or self.sig_B[parameter] == None:
                print('Error, first parameterize the sigmoid')
                exit(-1)
            y = self.sig_K / (1 + self.sig_Q[parameter] * math.exp(- self.sig_B[parameter] * x))
        # Exponential map
        else:
            if self.exp_A[parameter] == None or self.exp_B[parameter] == None:
                print('Error, first parameterize the exponential')
                exit(-1)
            y = self.exp_A[parameter] * math.exp( self.exp_B[parameter] * x )
        return y
        
    
# Adaptive elitism level ACOr
class AELACOr(SRAACOr):
    """ Adaptive control of the q parameter """
    def __init__(self):
        """ Constructor """
        super().__init__()
    
    def set_parameters(self, pop_size, k, xi, min_q, max_q, map_type, function_evaluations_array):
        """ Define values for the parameters used by the algorithm """
        # Input error checking
        if min_q > max_q:
            print('Error, maximum q must be greater than minimum q')
            exit(-1)
        if min_q <= 0:
            print('Error, minimum q must be greater than zero')
            exit(-1)
        if not isinstance(map_type, str):
            print('Error, map from success rate to q must be a string')
            exit(-1)
        if map_type != 'lin' and map_type != 'sig' and map_type != 'exp':
            print('Error, map type must be \'lin\', \'sig\' or \'exp\'')
            exit(-1)
        if map_type == 'sig' and max_q >= self.sig_K:
            print('Error, maximum q must be lesser than sigmoid K = ' + str(self.sig_K))
        
        # Parameter setting from ACOr class
        super().set_parameters(pop_size, k, max_q, xi, function_evaluations_array)    

        # Parameterize control curve
        self.min['q'] = min_q
        self.max['q'] = max_q
        self.map_type['q'] = map_type
        self.parameterize_map('q')
        
    
    def control_q(self):
        """ Use population success rate to update q """
        if self.success_rate == None:
            print("Error, compute success rate before updating q")
            exit(-1)
        
        # Compute new q, directly proportional (linearity or not) to the success rate
        self.q = self.evaluate_map('q', self.success_rate)
        
    
# Adaptive generation dispersion ACOr
class AGDACOr(SRAACOr):
    """ Adaptive control of the xi parameter """
    
    def __init__(self):
        """ Constructor """
        super().__init__()
    
    def set_parameters(self, pop_size, k, q, min_xi, max_xi, map_type, function_evaluations_array):
        """ Define values for the parameters used by the algorithm """
        # Input error checking
        if min_xi > max_xi:
            print('Error, maximum xi must be greater than minimum xi')
            exit(-1)
        if min_xi <= 0:
            print('Error, minimum xi must be greater than zero')
            exit(-1)
        if not isinstance(map_type, str):
            print('Error, map from success rate to xi must be a string')
            exit(-1)
        if map_type != 'lin' and map_type != 'sig' and map_type != 'exp':
            print('Error, map type must be \'lin\', \'sig\' or \'exp\'')
            exit(-1)
        if map_type == 'sig' and max_xi >= self.sig_K:
            print('Error, maximum xi must be lesser than sigmoid K = ' + str(self.sig_K))
        
        # Parameter setting from ACOr class
        super().set_parameters(pop_size, k, q, max_xi, function_evaluations_array)    

        # Minimum and maximum of adaptive xi
        # Parameterize control curve
        self.min['xi'] = min_xi
        self.max['xi'] = max_xi
        self.map_type['xi'] = map_type
        self.parameterize_map('xi')
        
    def control_xi(self):
        """ Use population success rate to update Xi """
        if self.success_rate == None:
            print("Error, compute success rate before updating xi")
            exit(-1)
        
        # Compute new xi, inversely proportional (linearity or not) to the success rate
        self.xi = self.evaluate_map('xi', (1 - self.success_rate))

    
# Bi-adaptive ACOr
class BAACOr(SRAACOr):
    """ Adaptive control of the both q and xi parameters """
    
    def __init__(self):
        """ Constructor """
        super().__init__()

    
    def set_parameters(self, pop_size, k, min_q, max_q, min_xi, max_xi, q_map_type, xi_map_type, function_evaluations_array):
        """ Define values for the parameters used by the algorithm """
        # Input error checking
        if min_xi > max_xi or min_q > min_q:
            print('Error, maximum parameters must be greater than minimum ones')
            exit(-1)
        if min_xi <= 0 or min_q <= 0:
            print('Error, minimum parameters must be greater than zero')
            exit(-1)
        if not isinstance(q_map_type, str) or not isinstance(xi_map_type, str):
            print('Error, maps from success rate to parameters must be strings')
            exit(-1)
        if  (q_map_type  != 'lin' and q_map_type  != 'sig' and q_map_type  != 'exp') or (xi_map_type != 'lin' and xi_map_type != 'sig' and xi_map_type != 'exp'):
            print('Error, map types must be \'lin\', \'sig\' or \'exp\'')
            exit(-1)
        if (q_map_type == 'sig' and max_q >= self.sig_K) or (xi_map_type == 'sig' and max_xi >= self.sig_K):
            print('Error, maximum parameters value must be lesser than sigmoid K = ' + str(self.sig_K))
            
        # Parameter setting from ACOr class
        super().set_parameters(pop_size, k, max_q, max_xi, function_evaluations_array)

        # Parameterize xi control curve
        self.min['xi'] = min_xi
        self.max['xi'] = max_xi
        self.map_type['xi'] = xi_map_type
        self.parameterize_map('xi')
        # Parameterize q control curve
        self.min['q'] = min_q
        self.max['q'] = max_q
        self.map_type['q'] = q_map_type
        self.parameterize_map('q')
        
    
    def control_xi(self):
        """ Use population success rate to update Xi """
        if self.success_rate == None:
            print("Error, first compute success rate")
            exit(-1)
        
        # Compute new xi
        self.xi = self.evaluate_map('xi', (1 - self.success_rate))
        
      
    def control_q(self):
        """ Use population success rate to update Xi """
        if self.success_rate == None:
            print("Error, first compute success rate")
            exit(-1)
        
        # Compute new q
        self.q = self.evaluate_map('q', self.success_rate)
       