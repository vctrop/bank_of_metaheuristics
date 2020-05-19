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

# Python standard lib
import sys
# 3rd party
import numpy as np
from deap.benchmarks import bohachevsky, rastrigin, cigar, sphere, schaffer 
# Own
from ant_colony_for_continuous_domains  import ACOr, ACSACOr, AGDACOr, MAACOr
from particle swarm optimization        import AIWPSO
from simulated_annealing                import ACFSA


if len(sys.argv) != 2:
    print('Please, run as \'%s {metaheuristic}\'' % (sys.argv[0]))
    exit(-1)
    
metaheuristic_str = sys.argv[1]
 
def parameterize_metaheuristic(metaheuristic_name, function_evals_array)
    if len(function_evals_array) == 0:
        print('Error, function evaluations array must not be empty')
        exit(-1)
    
    metaheuristic = None
    # ACOr
    if metaheuristic_name.lower() == 'acor':
        k = 50
        m = 10    
        q = 1e-4
        xi = 0.85
        
        # Number of function evaluations for ACOr: pop_size * num_iterations
        #num_iterations = (np.max(function_evals_array) - k) / m
        #print("# iterations = %d" % num_iterations) 
        #if not (num_iterations.is_integer()):
        #   print("Error, number of function evaluations subtracted by k is not divisible by population size")
        #   exit(-1)
        
        metaheuristic = ACOr()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(m,  k, q, xi, function_evals_array)
    
    # ACSACOr
    else if metaheuristic_name.lower() == 'acsacor':
        
        k = 50
        m = 10    
        xi = 0.85
        
        q_dict = np.load('./results/linear_nonlinear/linear_ACS.npy')
        q_min = q_dict.item()['min']
        q_max = q_min + q_dict.item()['max_minus_min']
        if q_max >= 1:
            q_max = 0.99
        
        # Number of function evaluations for ACOr: pop_size * num_iterations
        # num_iterations = (np.max(function_evals_array) - k) / m
        # print("# iterations = %d" % num_iterations) 
        # if not (num_iterations.is_integer()):
            # print("Error, number of function evaluations subtracted by k is not divisible by population size")
            # exit(-1)
        
        metaheuristic = ACSACOr()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(m, k, xi, q_min, q_max, True, function_evals_array)
    
    # AGDACOr
    else if metaheuristic_name.lower() == 'agdacor':
        
        k = 50
        m = 10    
        q = 1e-4
        xi_dict = np.load('./results/linear_nonlinear/linear_AGD.npy')
        xi_min = xi_dict.item()['min']
        xi_max = xi_min + xi_dict.item()['max_minus_min']
        if xi_max >= 1:
            xi_max = 0.99
        # Number of function evaluations for ACOr: pop_size * num_iterations
        # num_iterations = (np.max(function_evals_array) - k) / m
        # print("# iterations = %d" % num_iterations) 
        # if not (num_iterations.is_integer()):
            # print("Error, number of function evaluations subtracted by k is not divisible by population size")
            # exit(-1)
        
        metaheuristic = AGDACOr()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(m, k, q, xi_min, xi_max, True, function_evals_array)
    
    # MAACOr
    else if metaheuristic_name.lower() == 'maacor':
        
        k = 50
        m = 10    
        q_dict = np.load('./results/linear_nonlinear/linear_ACS.npy')
        q_min = q_dict.item()['min']
        q_max = q_min + q_dict.item()['max_minus_min']
        if q_max >= 1:
            q_max = 0.99
        xi_dict = np.load('./results/linear_nonlinear/linear_AGD.npy')
        xi_min = xi_dict.item()['min']
        xi_max = xi_min + xi_dict.item()['max_minus_min']
        if xi_max >= 1:
            xi_max = 0.99
        
        # Number of function evaluations for ACOr: pop_size * num_iterations
        # num_iterations = (np.max(function_evals_array) - k) / m
        # print("# iterations = %d" % num_iterations) 
        # if not (num_iterations.is_integer()):
            # print("Error, number of function evaluations subtracted by k is not divisible by population size")
            # exit(-1)    
        
        metaheuristic = MAACOr()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(m, k, q_min, q_max, xi_min, xi_max, True, True, function_evals_array)
    
    # SA
    else if metaheuristic_name.lower() == 'sa':
        local_iterations = 
        initial_temperature = 
        cooling_constant = 
        step_size = 
        local_iterations = 
        # Number of function evaluations for SA: global_iterations * local_iterations
        # global_iterations = np.max(function_evals_array) / local_iterations
        # print("# local/global iterations = %d/%d" % (local_iterations, global_iterations)) 
        # if not (global_iterations.is_integer()):
            # print("Error, number of function evaluations is not divisible by number of local iterations")
            # exit(-1)
            
        metaheuristic = SA()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(initial_temperature, cooling_constant, step_size, local_iterations, function_evals_array)
    
    # ACFSA
    else if metaheuristic_name.lower() == 'acfsa':
        local_iterations = 
        initial_temperature = 
        cooling_constant = 
        
        # Number of function evaluations for SA: global_iterations * local_iterations
        # global_iterations = np.max(function_evals_array) / local_iterations
        # print("# local/global iterations = %d/%d" % (local_iterations, global_iterations)) 
        # if not (global_iterations.is_integer()):
            # print("Error, number of function evaluations is not divisible by number of local iterations")
            # exit(-1)
        
        metaheuristic = ACFSA()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(initial_temperature, cooling_constant, local_iterations, function_evals_array)
   
   # PSO
    else if metaheuristic_name.lower() == 'pso':

        population_size = 
        personal_acceleration = 
        global_acceleration = 
        
        # Number of function evaluations for PSO: swarm_size * num_iterations
        # num_iterations = np.max(function_evals_array) / population_size
        # print("# iterations = %d" % num_iterations) 
        # if not (num_iterations.is_integer()):
            # print("Error, number of function evaluations is not divisible by swarm size")
            # exit(-1)
            
        metaheuristic = PSO()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(population_size, personal_acceleration, global_acceleration, function_evals_array)
    
    # AIWPSO
    else if metaheuristic_name.lower() == 'aiwpso':
        population_size = 
        personal_acceleration = 
        global_acceleration =
        min_inertia = 
        max_inertia =
        
        # Number of function evaluations for PSO: swarm_size * num_iterations
        # num_iterations = np.max(function_evals_array) / population_size
        # print("# iterations = %d" % num_iterations) 
        # if not (num_iterations.is_integer()):
            # print("Error, number of function evaluations is not divisible by swarm size")
            # exit(-1)
        
        metaheuristic = AIWPSO()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(population_size, personal_acceleration, global_acceleration, min_inertia, max_inertia, function_evals_array)
    else:
        print('Error, the chosen metaheuristic is not yet supported for this experiment')
        exit(-1)
        
    return metaheuristic
    
def run_metaheuristic_test_functions():
    test_functions = [bohachevsky, cigar, rastrigin, schaffer, sphere]
    functions_names = ['bohachevsky', 'cigar', 'rastrigin', 'schaffer', 'sphere']
    function_evaluations = [10000 * i for i in range(1,21)]     # Establish 20 function evaluations of interested, uniformly from 10k to 200k
    functions_bounding = {  'bohachevsky':  True,
                            'cigar':        False, 
                            'rastrigin':    True, 
                            'schaffer':     True, 
                            'sphere':       False}
    functions_ranges = {    'bohachevsky':  [-100   , 100],  
                            'cigar':        [-10    , 10],      # unbounded, values used in initialization only
                            'rastrigin':    [-5.12  , 5.12],  
                            'schaffer':     [-100   , 100],    
                            'sphere':       [-10    , 10]}      # unbounded, values used in initialization only

    num_runs = 100

    # For all test objective functions, run the given metaheuristic for a number of times
    for function, function_str in zip(test_functions, functions_names):
        # Get metaheuristic object with parameters already defined
        metaheuristic = parameterize_metaheuristic(metaheuristic_str, function_evaluations)
        # Configure search space of the objective function
        variables_bounded = functions_bounding[function_str]
        variables_range = function_ranges[function_str]
        dimensionality = 2                     # Number of variables for all functions
        ranges      = [variables_range  for _ in range(dimensionality)]
        is_bounded  = [bounded          for _ in range(dimensionality)]
        # Update the metaheuristic with objective function information
        metaheuristic.define_variables(ranges, is_bounded)
        metaheuristic.set_cost(function)
        
        costs_matrix = []
        for _ in range(num_runs):
            # An optimization call returns the solutions found during optimization for the specified function evaluation values in parameterization
            solutions_at_FEs = metaheuristic.optimize()
            costs_array = solutions_at_FEs[:, -1]
            costs_matrix.append(costs_array)
        np.save('./results/metaheuristics_comparison/' + function_str + '_' + metaheuristic_str + '.npy','wb')


if __name__ == '__main__':
    