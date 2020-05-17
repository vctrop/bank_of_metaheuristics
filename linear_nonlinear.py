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
# Own
from ant_colony_for_continuous_domains import ACSACOr, AGDACOr
## 3rd party
import numpy as np
# Benchmarking functions
from deap.benchmarks import rosenbrock, schwefel, ackley, griewank, himmelblau      # Load functions used as train instances
# SMAC
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade  import SMAC4HPO
from smac.scenario.scenario import Scenario




# Given a function, an adaptive mechanism for ACOr, parameters and linearity of adaptive parameter control,
#  return the average results of a number runs on a suite of functions
def function_cost(function, range, bounded, acor_mechanism, minimum, maximum, linear_control):
    if not isinstance(acor_mechanism, str):
        print("Error, ACOr mechanism must be a string")
        exit(-1)
    if acor_mechanism.upper() != "ACS" and acor_mechanism.upper() != "AGD":
        print("Error, ACOr mechanism must be either \"ACS\" or \"AGS\"")
        exit(-1)
    if not isinstance(linear_control, bool):
        print("Error, linear control must be a boolean")
        exit(-1)
        
    # Base ACOr parameters, from (Socha, 2008)
    # Number of function evaluations (F.E.) = k + iterations * m
    iterations = 75    # 75 iterations = 200 F.E.
    k = 50
    m = 2    
    q = 1e-4
    xi = 0.85
    
    if acor_mechanism == "ACS":
        colony = ACSACOr()
        colony.set_verbosity(False)
        colony.set_parameters(iterations, m, k, xi, minimum, maximum, linear_control)
    else:
        colony = AGDACOr()  
        colony.set_verbosity(False)
        colony.set_parameters(iterations, m, k, q, minimum, maximum, linear_control)
    
    # Define ranges and bounding of each variable
    dimensionality = 2                     # Number of variables for all functions
    ranges      = [range    for _ in range(dimensionality)]
    is_bounded  = [bounded  for _ in range(dimensionality)]
    colony.define_variables(ranges, is_bounded)
    
    # Get cost for the given function
    colony.set_cost(function)
    function_cost = colony.optimize()[-1]
        
    return function_cost
    

def define_smac_cost(function, variables_range, variables_bounded, acor_mechanism, linear_control):
    def cost(smac_solution):
        # The SMAC solution stores None-values for deactivated parameters, and we remove them here
        smac_solution = {k: smac_solution[k] for k in smac_solution if smac_solution[k]}
        minimum = smac_solution['min']
        maximum = minimum + smac_solution['max_minus_min']
        if maximum >= 1:
            maximum = 0.99
        
        return function_cost(function, variables_range, variables_bounded, acor_mechanism, minimum, maximum, linear_control)
        
    return cost
    
    
train_functions = [rosenbrock, schwefel, ackley, griewank, himmelblau]
train_functions_names = ['rosenbrock', 'schwefel','ackley','griewank','himmelblau']
mechanisms = ['ACS', 'AGD']               
num_smac = 2
function_bounding = {'rosenbrock':  False,
                     'schwefel':    True, 
                     'ackley':      True, 
                     'griewank':    True, 
                     'himmelblau':  True}
                        
function_ranges = {'rosenbrock':    [-10, 10],   # unbounded, initialization only
                   'schwefel':      [-500, 500],
                   'ackley':        [-15, 30],  
                   'griewank':      [-600, 600],
                   'himmelblau':    [-6,6]}     
                    
mechanism_ranges = {'ACS':[1e-8, 1e-1],
                    'AGD':[0.01, 0.99]}
                    
# Run SMAC a number of times searching parameters (adaptive minima and maxima) for ACS and AGD in all train functions, using linear and nonlinear adaptive parameter control
for objective_function, function_str in zip(train_functions, train_functions_names):
    variables_bounded   = function_bounding[function_str]
    variables_range     = function_ranges[function_str]
    
    for mechanism in mechanisms:
        parameter_range = mechanism_ranges[mechanism]
        
        for use_linear, linearity_str in zip([True, False], ['true', 'false']):
            smac_solutions = []
            
            for _ in range(num_smac):
                ## Run SMAC for the given mechanism in a given dataset
                # Build Configuration Space which defines all parameters and their ranges
                cs = ConfigurationSpace()
                min =           UniformFloatHyperparameter("min"            , parameter_range[0], parameter_range[1], default_value=(parameter_range[0]+parameter_range[1])/2)
                max_minus_min = UniformFloatHyperparameter("max_minus_min"  , parameter_range[0], parameter_range[1], default_value=(parameter_range[0]+parameter_range[1])/2)
                cs.add_hyperparameters([min, max_minus_min])
                # Scenario object
                scenario = Scenario({"run_obj": "quality",      # Optimize quality instead of runtime
                                     "runcount-limit": 10,      # max. number of function evaluations
                                     "cs": cs,                  # configuration space
                                     "deterministic": "false"   # each metaheuristic being optimized is stochastic
                                     })

                # Optimize, using a SMAC-object
                smac = SMAC4HPO(scenario    = scenario,
                                rng         = np.random.RandomState(42),
                                tae_runner  = define_smac_cost(objective_function, variables_range, variables_bounded, mechanism, use_linear))

                solution = smac.optimize()
                
        
        np.save('./results/linear_nonlinear/' + linearity_str + '_'+ mechanism + '_' + function_str + '.npy', smac_solutions)