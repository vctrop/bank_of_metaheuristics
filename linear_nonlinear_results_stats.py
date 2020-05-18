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

# 3rd party
import numpy as np
from scipy.stats import wilcoxon
# Own
import linear_nonlinear_experiment

# Print average and standard deviation of all metaheuristic applications for a given SMAC solution 
def print_avg_std():
    mechanisms = ['ACS', 'AGD']
    function_names = ['rosenbrock', 'schwefel','ackley','griewank','himmelblau']
    
    for mechanism in mechanisms:
        print('[' + mechanism + 'ACOr]')
        
        for function_str in function_names:  
            print(function_str.upper())
        
            for linearity_str in ['linear', 'nonlinear']:
                print(linearity_str)
                function_costs = np.load('./results/linear_nonlinear/' + linearity_str + '_'+ mechanism + '_' + function_str + '_eval.npy')
                mean = np.mean(function_costs)
                std  = np.std(function_costs)
                print(str(format(mean, '.3E')) + ' (' + str(format(std, '.3E')) +')')
            print('\n')
            
# Print average and standard deviation of all metaheuristic applications for a given SMAC solution 
def print_p_values():
    mechanisms = ['ACS', 'AGD']
    function_names = ['rosenbrock', 'schwefel','ackley','griewank','himmelblau']
    
    for mechanism in mechanisms:
        print('[' + mechanism + 'ACOr]')
        
        for function_str in function_names:  
            print(function_str.upper())
            
            linear_costs    = np.load('./results/linear_nonlinear/linear' + '_'+ mechanism + '_' + function_str + '_eval.npy')
            nonlinear_costs = np.load('./results/linear_nonlinear/nonlinear' + '_'+ mechanism + '_' + function_str + '_eval.npy')
            _, pval = wilcoxon(linear_costs, nonlinear_costs)
            print('p-value = ' + str(format(pval, '.5f')) + '\n')
            

print_avg_std()
print_p_values()

