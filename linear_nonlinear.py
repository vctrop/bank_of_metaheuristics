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
# 3rd party
import numpy as np
# Own
from ant_colony_for_continuous_domains import ACSACOr, AGDACOr
from deap.benchmarks import ackley

# Given an adaptive mechanism for ACOr, linearity of parameter control and adaption parameters,
#  return the average results of a number runs on a suite of functions
def average_cost(num_iterations, acor_mechanism, minimum, maximum, linear_control, function_list):
    if not isinstance(acor_mechanism, str):
        print("Error, ACOr mechanism must be a string")
        exit(-1)
    if acor_mechanism.upper() != "ACS" and acor_mechanism.upper() != "AGD":
        print("Error, ACOr mechanism must be either \"ACS\" or \"AGS\"")
        exit(-1)
    if not isinstance(linear_control, bool):
        print("Error, linear control must be a boolean")
        exit(-1)
    if len(function_list) == 0:
        print("Error, function list must not be empty")
        exit(-1)
        
    # Base ACOr parameters, from (Socha, 2008)
    k = 50
    pop_size = 2
    q = 1e-4
    xi = 0.85
    ranges = [[-5,5],
              [-5,5],
              [-5,5]]

    is_bounded = [True, True, True]
     
    if acor_mechanism == "ACS":
        colony = ACSACOr()
        colony.set_verbosity(False)
        colony.set_parameters(num_iterations, pop_size, k, xi, minimum, maximum, linear_control)
        colony.define_variables(ranges, is_bounded)
    else:
        colony = AGDACOr()  
        colony.set_verbosity(False)
        colony.set_parameters(num_iterations, pop_size, k, q, minimum, maximum, linear_control)
        colony.define_variables(ranges, is_bounded)
    
    # Sum of function costs
    total_cost = 0.0
    for objective_function in function_list:
        colony.set_cost(objective_function)
        solution_cost = colony.optimize()[-1]
        total_cost += solution_cost
        
    return total_cost / len(function_list)
    
print(average_cost(1000, "ACS", 0.2, 0.8, False, [ackley]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    