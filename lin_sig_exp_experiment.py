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

# Python standard library
import math
import sys
import io
# Own
from ant_colony_for_continuous_domains import AELACOr, AGDACOr
## 3rd party
import numpy as np
# Benchmarking functions
from deap.benchmarks import rosenbrock, schwefel, ackley, griewank, himmelblau      # Load functions used as train instances

def flatten_cost(cost_function):
    def flattened_cost(x):
        return cost_function(x)[0]
    return flattened_cost

def function_cost(function, variables_range, bounded, adaptive_mechanism, map_type, function_evals):
    """ Given a function, an adaptive mechanism for ACOr, parameters and linearity of adaptive parameter control,
        return the average results of a number runs on a suite of functions."""
    if not isinstance(adaptive_mechanism, str) or not isinstance(map_type, str):
        print("Error, both adaptive mechanism and map type must be strings")
        exit(-1)
    if adaptive_mechanism.upper() != "AEL" and adaptive_mechanism.upper() != "AGD":
        print("Error, ACOr mechanism must be either \"AEL\" or \"AGS\"")
        exit(-1)
    if map_type != 'lin' and map_type != 'sig' and map_type != 'exp':
        print("Error, map type must be \"lin\", \"sig\" or \"exp\"")
        exit(-1)
        
    # Base ACOr parameters, from (Socha, 2008)
    # Number of function evaluations (F.E.) = k + iterations * m
    k = 50
    m = 10    
    q = 1e-4
    xi = 0.85
    
    if adaptive_mechanism == "AEL":
        colony = AELACOr()
        colony.set_verbosity(False)
        min_q = 1e-4
        max_q = 1.0
        colony.set_parameters(m, k, xi, min_q, max_q, map_type, [function_evals])
    else:
        colony = AGDACOr()  
        colony.set_verbosity(False)
        min_xi = 0.1
        max_xi = 0.93
        colony.set_parameters(m, k, q, min_xi, max_xi, map_type, [function_evals])
    
    # Define ranges and bounding of each variable
    dimensionality = 3                     # Number of variables for all functions
    ranges      = [variables_range  for _ in range(dimensionality)]
    is_bounded  = [bounded          for _ in range(dimensionality)]
    colony.define_variables(ranges, is_bounded)
    
    # Get cost for the given function
    colony.set_cost(flatten_cost(function))
    function_cost = colony.optimize()[-1][-1]
        
    return function_cost

    
def run_lin_sig_exp_mappings():    
    """ Run AELACOr and AGDACOr in 4 train functions, each using linear, sigmoidal and exponential mappings from SR to parameters. """
    
    train_functions =       [rosenbrock  , schwefel  , ackley , griewank]
    train_functions_names = ['rosenbrock', 'schwefel','ackley','griewank']
    functions_bounding =    {'rosenbrock':  False,
                            'schwefel':    True, 
                            'ackley':      True, 
                            'griewank':    True}
    functions_ranges =      {'rosenbrock':   [-10 , 10],   # unbounded, values used in initialization only
                            'schwefel':      [-500, 500],
                            'ackley':        [-15 , 30],  
                            'griewank':      [-600, 600]} 
    
    metaheuristic_runs = 100
    for map_type in (['lin', 'sig', 'exp']):
        for mechanism in ['AEL', 'AGD'] :
            metaheuristic_function_evals = 10000
            
            # For each train function, run a metaheuristic N times and save results
            for function, function_str in zip(train_functions, train_functions_names):        
                variables_bounded = functions_bounding[function_str]
                variables_range = functions_ranges[function_str]
                
                # Run metaheuristic N times
                function_costs = []
                for i in range(metaheuristic_runs):
                    cost = function_cost(function, variables_range, variables_bounded, mechanism, map_type, metaheuristic_function_evals)
                    function_costs.append(cost)
                    print(str(i) + '. ' + str(cost))
                np.save('./results/linear_nonlinear/' + map_type + '_'+ mechanism + '_' + function_str + '_eval.npy', function_costs)
                

if __name__ == '__main__':
    run_lin_sig_exp_mappings() 
