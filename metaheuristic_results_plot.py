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
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print('Error, please use: python3 %s {objective_function}' % (sys.argv[0]))
    exit(-1)
    
function_name = (sys.argv[1]).lower()
test_functions_names = ['bohachevsky', 'cigar', 'rastrigin', 'schaffer', 'sphere', 'himmelblau']
if not (function_name in test_functions_names):
    print('Error, results are available for test functions only (bohachevsky, cigar, rastrigin, schaffer, sphere and himmelblau)')
    exit(-1)
    
function_evaluations = [100 * i for i in range(1, 100)] + [10000 * i for i in range(1,11)] 
metaheuristics_names = ['acor', 'aelacor', 'agdacor', 'baacor', 'aiwpso', 'acfsa']
colors               = []
plt.figure()    
for index, metaheuristic_str in enumerate(metaheuristics_names):
    costs_matrix = np.load('./results/metaheuristics_comparison/' + function_name + '_' + metaheuristic_str + '_costs.npy')
    average_cost_trajectory = np.sum(costs_matrix, axis = 0)
    average_cost_trajectory /= 30
    plt.plot(function_evaluations, average_cost_trajectory, label=metaheuristic_str, linewidth=3)

plt.xlabel('Function evaluations', fontsize=18)
plt.ylabel('Cost', fontsize=18)
plt.legend(fontsize=16)

plt.show()