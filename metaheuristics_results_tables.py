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
import sys
# 3rd party
import numpy as np
from scipy.stats import wilcoxon

metaheuristics_names = ['acor', 'aelacor', 'agdacor', 'baacor', 'aiwpso', 'acfsa']
test_functions_names = ['bohachevsky', 'cigar', 'rastrigin', 'schaffer', 'sphere', 'himmelblau']

print('COST AND TIMING COMPARISON')
for function_str in test_functions_names:
    print('[ ' + function_str + ' ]')
    # pval_aux_matrix = []
    for index, metaheuristic_str in enumerate(metaheuristics_names):
        costs_matrix = np.load('./results/metaheuristics_comparison/' + function_str + '_' + metaheuristic_str + '_costs.npy')
        last_iter_costs = costs_matrix[:, -1]
        # print(np.shape(last_iter_costs))
        
        avg_cost = np.mean(last_iter_costs)
        std_cost = np.std(last_iter_costs)
        print(metaheuristic_str.upper())
        print('Cost: ' + str(format(avg_cost,'.3E')) + ' (' + str(format(std_cost,'.3E')) + ')')
        # pval_aux_matrix.append([avg_cost, list(last_iter_costs), metaheuristic_str])
        
        times_array = np.load('./results/metaheuristics_comparison/' + function_str + '_' + metaheuristic_str + '_times.npy')
        # print(np.shape(times_array))
        avg_time = np.mean(times_array)
        print('Time: ' + str(format(avg_time,'.3E')))
    # Sort p-val auxiliary matrix according to the averages
    # pval_aux_matrix = np.array(pval_aux_matrix)
    # pval_aux_matrix = pval_aux_matrix[pval_aux_matrix[:, 0].argsort()]             
    # # Compute p-val between the arrays of the best and second best approaches   
    # if not ((np.array(pval_aux_matrix[0,1]) - np.array(pval_aux_matrix[1,1])) == 0).all():
        # _, pval = wilcoxon(pval_aux_matrix[0,1], pval_aux_matrix[1,1])
        # print('(p-value) ' + str(pval_aux_matrix[0,2]) + ' v ' + str(pval_aux_matrix[1,2]) + ' = ' + str(format(pval,'.3')))    
    # else:
        # print('p-value unavailable')
    print('\n')

    
    
print('\n\nSTATTISTICAL SIGNIFICANCE BETWEEN ACOr VARIATIONS')
acor_mechanisms = ['ael', 'agd', 'ba']
for function_str in test_functions_names:
    print('[ ' + function_str + ' ]')
    acor_costs_matrix = np.load('./results/metaheuristics_comparison/' + function_str + '_' + 'acor_costs.npy')
    acor_last_costs = acor_costs_matrix[:, -1]
    
    for mechanism in acor_mechanisms:    
        mech_costs_matrix = np.load('./results/metaheuristics_comparison/' + function_str + '_' + mechanism + 'acor_costs.npy')
        mech_last_costs = mech_costs_matrix[:, -1]
        if not (mech_last_costs - acor_last_costs == 0).all():
            _, pval = wilcoxon(acor_last_costs, mech_last_costs)
            print('ACOr v ' + mechanism.upper() + 'ACOr: ' + str(format(pval,'.4f')))
        else:
            print('ACOr v ' + mechanism.upper() + 'ACOr: Unavailable')
        
        
    