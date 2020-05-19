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
from particle_swarm_optimization        import PSO, AIWPSO
from simulated_annealing                import SA, ACFSA

if len(sys.argv) != 2:
    print('Please, run as \'%s {metaheuristic}\'' % (sys.argv[0]))
    exit(-1)
    
metaheuristic_str = sys.argv[1]
 
def parameterize_metaheuristic(metaheuristic_name, function_evals_array):
    if len(function_evals_array) == 0:
        print('Error, function evaluations array must not be empty')
        exit(-1)
    
    metaheuristic = None
    # ACOr
    if metaheuristic_name.lower() == 'acor':
        # Parameters
        k = 50; m = 10; q = 1e-4; xi = 0.85
        # Configure
        metaheuristic = ACOr()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(m,  k, q, xi, function_evals_array)
    
    # ACSACOr
    elif metaheuristic_name.lower() == 'acsacor':
        # Parameters
        k = 50; m = 10; xi = 0.85
        q_dict = np.load('./results/linear_nonlinear/nonlinear_ACS.npy')
        q_min = q_dict.item()['min']
        q_max = q_min + q_dict.item()['max_minus_min']
        if q_max >= 1:
            q_max = 0.99
        # Configure
        metaheuristic = ACSACOr()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(m, k, xi, q_min, q_max, False, function_evals_array)
    
    # AGDACOr
    elif metaheuristic_name.lower() == 'agdacor':
        # Parameters
        k = 50; m = 10; q = 1e-4; 
        xi_dict = np.load('./results/linear_nonlinear/nonlinear_AGD.npy')
        xi_min = xi_dict.item()['min']
        xi_max = xi_min + xi_dict.item()['max_minus_min']
        if xi_max >= 1:
            xi_max = 0.99
        # Configure
        metaheuristic = AGDACOr()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(m, k, q, xi_min, xi_max, False, function_evals_array)
    
    # MAACOr
    elif metaheuristic_name.lower() == 'maacor':
        # Parameters
        k = 50; m = 10
        q_dict = np.load('./results/linear_nonlinear/nonlinear_ACS.npy')
        q_min = q_dict.item()['min']
        q_max = q_min + q_dict.item()['max_minus_min']
        if q_max >= 1:
            q_max = 0.99
        xi_dict = np.load('./results/linear_nonlinear/nonlinear_AGD.npy')
        xi_min = xi_dict.item()['min']
        xi_max = xi_min + xi_dict.item()['max_minus_min']
        if xi_max >= 1:
            xi_max = 0.99
        # Configure
        metaheuristic = MAACOr()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(m, k, q_min, q_max, xi_min, xi_max, False, False, function_evals_array)
    
    # SA
    elif metaheuristic_name.lower() == 'sa':
        # Parameters
        local_iterations = 500
        initial_temperature = 50
        cooling_constant = 0.99
        step_size = 1e-2
        # Configure
        metaheuristic = SA()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(initial_temperature, cooling_constant, step_size, local_iterations, function_evals_array)
    
    # ACFSA
    elif metaheuristic_name.lower() == 'acfsa':
        # Parameters
        local_iterations = 500
        initial_temperature = 50
        cooling_constant = 0.99 
        # Configure
        metaheuristic = ACFSA()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(initial_temperature, cooling_constant, local_iterations, function_evals_array)
   
   # PSO
    elif metaheuristic_name.lower() == 'pso':
        # Parameters
        population_size = 20
        personal_acceleration = 2
        global_acceleration = 2
        # Configure
        metaheuristic = PSO()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(population_size, personal_acceleration, global_acceleration, function_evals_array)
    
    # AIWPSO
    elif metaheuristic_name.lower() == 'aiwpso':
        # Parameters
        population_size = 20
        personal_acceleration = 2
        global_acceleration = 2
        min_inertia = 0.0
        max_inertia = 1.0
        # Configure
        metaheuristic = AIWPSO()
        metaheuristic.set_verbosity(False)
        metaheuristic.set_parameters(population_size, personal_acceleration, global_acceleration, min_inertia, max_inertia, function_evals_array)
    else:
        print('Error, the chosen metaheuristic is not yet supported for this experiment')
        exit(-1)
        
    return metaheuristic
    
def run_metaheuristic_test_functions(metaheuristic_name):
    test_functions = [bohachevsky, cigar, rastrigin, schaffer, sphere]
    functions_names = ['bohachevsky', 'cigar', 'rastrigin', 'schaffer', 'sphere']
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
    
    # Establish 40 function evaluations of interest, uniformly from 5k to 200k
    function_evaluations = [5000 * i for i in range(1,41)]     
    #function_evaluations = [100 * i for i in range(1,21)]     
    # Number of times each metaheuristic will run in each function
    num_runs = 100
    #num_runs = 5

    # For all test objective functions, run the given metaheuristic for a number of times
    for function, function_str in zip(test_functions, functions_names):
        # Get metaheuristic object with parameters already defined
        metaheuristic = parameterize_metaheuristic(metaheuristic_name, function_evaluations)
        # Configure search space of the objective function
        variables_bounded = functions_bounding[function_str]
        variables_range = functions_ranges[function_str]
        dimensionality = 2                     # Number of variables for all functions
        ranges      = [variables_range   for _ in range(dimensionality)]
        is_bounded  = [variables_bounded for _ in range(dimensionality)]
        # Update the metaheuristic with objective function information
        metaheuristic.define_variables(ranges, is_bounded)
        metaheuristic.set_cost(function)
        
        costs_matrix = []
        for _ in range(num_runs):
            # An optimization call returns the solutions found during optimization for the specified function evaluation values in parameterization
            solutions_at_FEs = metaheuristic.optimize()
            costs_array = solutions_at_FEs[:, -1]
            costs_matrix.append(costs_array)
            print(costs_array)
        np.save('./results/metaheuristics_comparison/' + function_str + '_' + metaheuristic_name + '.npy', costs_matrix)


if __name__ == '__main__':
    run_metaheuristic_test_functions(metaheuristic_str)