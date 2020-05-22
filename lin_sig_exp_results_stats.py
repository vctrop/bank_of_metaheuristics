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
            pval_aux_matrix = []
            mapping_types = ['lin', 'sig', 'exp']
            for map_type in mapping_types:
                print(map_type)
                function_costs = np.load('./results/lin_sig_exp/' + map_type + '_'+ mechanism + '_' + function_str + '_eval.npy')
                mean = np.mean(function_costs)
                std  = np.std(function_costs)
                print(str(format(mean, '.3E')) + ' (' + str(format(std, '.3E')) +')')
                pval_aux_matrix.append([mean, list(function_costs), map_type])
            
            #if not (function_str == 'ackley' and mechanism == 'AEL'):
            # Sort p-val auxiliary matrix according to the averages
            pval_aux_matrix = np.array(pval_aux_matrix)
            pval_aux_matrix = pval_aux_matrix[pval_aux_matrix[:, 0].argsort()]             
            # Compute p-val between the arrays of the best and second best approaches   
            _, pval = wilcoxon(pval_aux_matrix[0,1], pval_aux_matrix[1,1])
            print('(p-value) ' + str(pval_aux_matrix[0,2]) + ' v ' + str(pval_aux_matrix[1,2]) + ' = ' + str(format(pval,'.3')))
            print('\n')

if __name__ == '__main__':
    print_avg_std()

