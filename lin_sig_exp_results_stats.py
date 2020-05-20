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

# Print average and standard deviation of all metaheuristic applications for a given SMAC solution 
def print_avg_std():
    function_names = ['rosenbrock', 'schwefel','ackley','griewank']
    
    for mechanism in ['AEL', 'AGD']:
        print('[' + mechanism + 'ACOr]')
        
        for function_str in function_names:  
            print(function_str.upper())
        
            for map_type in ['lin', 'sig', 'exp']:
                print(map_type)
                function_costs = np.load('./results/lin_sig_exp/' + map_type + '_'+ mechanism + '_' + function_str + '_eval.npy')
                mean = np.mean(function_costs)
                std  = np.std(function_costs)
                print(str(format(mean, '.3E')) + ' (' + str(format(std, '.3E')) +')')
            print('\n')
            
# Print average and standard deviation of all metaheuristic applications for a given SMAC solution 
def print_p_values():
    mechanisms = ['AEL', 'AGD']
    function_names = ['rosenbrock', 'schwefel','ackley','griewank']
    
    for mechanism in mechanisms:
        print('[' + mechanism + 'ACOr]')
        
        for function_str in function_names:  
            print(function_str.upper())
            
            lin_costs = np.load('./results/lin_sig_exp/lin' + '_'+ mechanism + '_' + function_str + '_eval.npy')
            sig_costs = np.load('./results/lin_sig_exp/sig' + '_'+ mechanism + '_' + function_str + '_eval.npy')
            exp_costs = np.load('./results/lin_sig_exp/exp' + '_'+ mechanism + '_' + function_str + '_eval.npy')
            
            if ?:
                _, pval = wilcoxon(linear_costs, nonlinear_costs)
            else:
                
            print('p-value = ' + str(format(pval, '.5f')) + '\n')
            

print_avg_std()
#print_p_values()

