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
import time
# 3rd party 
from deap.benchmarks import ackley, rastrigin
import numpy as np
import matplotlib.pyplot as plt
# Own
import acor_plots
if len(sys.argv) != 2:
    print("Please enter \"%s {number of function evaluations}\"" % sys.argv[0])
    exit(-1)
    
num_func_evals = int(sys.argv[1])
print("Num. FE = %d" % num_func_evals)

ranges = [[-5,5],
          [-5,5],
          [-5,5]]

is_bounded = [True, True, True]
     
def flatten_cost(cost_function):
    def flattened_cost(x):
        return cost_function(x)[0]
    return flattened_cost

# # ACOr
# # Total # of function evaluations: archive_size + population_size * num_iterations
# print("ACOr")
# # Parameters
# k = 50;  pop_size = 10;  q = 1e-2; xi = 0.85
# # Configure and run
# colony = acor_plots.ACOr()  
# colony.set_verbosity(False)
# colony.set_cost(flatten_cost(ackley))
# colony.set_parameters(pop_size, k, q, xi, [num_func_evals])
# colony.define_variables(ranges, is_bounded)
# solution = colony.optimize()
# print(solution)

# # AELACOr
# # Total # of function evaluations: archive_size + population_size * num_iterations
# print("AELACOr")
# # Parameters
# k = 50;  pop_size = 10; xi = 0.85
# min_q = 1e-2
# max_q = 1.0
# # Configure and run
# colony = acor_plots.AELACOr()
# colony.set_verbosity(False)
# colony.set_cost(flatten_cost(ackley))
# colony.set_parameters(pop_size, k, xi, min_q, max_q, 'exp', [num_func_evals])
# colony.define_variables(ranges, is_bounded)
# solution = colony.optimize()
# print(solution)


# # AGDACOr
# # Total # of function evaluations: archive_size + population_size * num_iterations
# print("AGDACOr")
# # Parameters
# k = 50;  pop_size = 10; q = 0.01
# min_xi = 0.1
# max_xi = 0.93
# # Configure and run
# colony = acor_plots.AGDACOr()  
# colony.set_verbosity(False)
# colony.set_cost(flatten_cost(ackley))
# colony.set_parameters(pop_size, k, q, min_xi, max_xi, 'sig', [num_func_evals])
# colony.define_variables(ranges, is_bounded)
# solution = colony.optimize()
# print(solution)


# BAACOr
print("BAACOr")
# Parameters
k = 50;  pop_size = 10;
min_q = 1e-2
max_q = 1.0   
min_xi = 0.1
max_xi = 0.93

# Configure and run
colony = acor_plots.BAACOr()  
colony.set_verbosity(False)
colony.set_cost(flatten_cost(rastrigin))
colony.set_parameters(pop_size, k, min_q, max_q, min_xi, max_xi, 'exp', 'sig', [num_func_evals])
colony.define_variables(ranges, is_bounded)
q_runs = []
xi_runs= []
for _ in range(100):
    qs, xis, solution = colony.optimize()
    q_runs.append(qs)
    xi_runs.append(xis)
    
np.shape(q_runs)
np.shape(xi_runs)

avg_q = np.sum(q_runs, axis=0) / len(q_runs)
avg_xi = np.sum(xi_runs, axis=0) / len(xi_runs)

plt.figure(figsize=(10,10))
plt.plot(avg_q, color = 'blue', label='avg_q')
plt.plot(avg_xi, color = 'orange', label='avg_xi')
plt.legend()
plt.show()