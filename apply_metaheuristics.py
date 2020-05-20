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
    
# Own
import ant_colony_for_continuous_domains
import particle_swarm_optimization
import simulated_annealing
from deap.benchmarks import ackley
import sys

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
   
# ACOr
# Total # of function evaluations: archive_size + population_size * num_iterations
print("ACOr")
# Parameters
k = 50;  pop_size = 2;  q = 0.01; xi = 0.85
# Configure and run
colony = ant_colony_for_continuous_domains.ACOr()  
colony.set_verbosity(False)
colony.set_cost(flatten_cost(ackley))
colony.set_parameters(pop_size, k, q, xi, [num_func_evals])
colony.define_variables(ranges, is_bounded)
solution = colony.optimize()
print(solution)

# AELACOr
# Total # of function evaluations: archive_size + population_size * num_iterations
print("AELACOr")
# Parameters
k = 50;  pop_size = 10; xi = 0.85
min_q = 1e-4
max_q = 1.0
# Configure and run
colony = ant_colony_for_continuous_domains.AELACOr()
colony.set_verbosity(False)
colony.set_cost(flatten_cost(ackley))
colony.set_parameters(pop_size, k, xi, min_q, max_q, 'lin', [num_func_evals])
colony.define_variables(ranges, is_bounded)
solution = colony.optimize()
print(solution)


# AGDACOr
# Total # of function evaluations: archive_size + population_size * num_iterations
print("AGDACOr")
# Parameters
k = 50;  pop_size = 10; q = 0.01
min_xi = 0.1
max_xi = 0.93
# Configure and run
colony = ant_colony_for_continuous_domains.AGDACOr()  
colony.set_verbosity(False)
colony.set_cost(flatten_cost(ackley))
colony.set_parameters(pop_size, k, q, min_xi, max_xi, 'lin', [num_func_evals])
colony.define_variables(ranges, is_bounded)
solution = colony.optimize()
print(solution)


# BAACOr
print("BAACOr")
# Parameters
k = 50;  pop_size = 10;
min_q = 1e-4  
max_q = 1.0   
min_xi = 0.1
max_xi = 0.93

# Configure and run
colony = ant_colony_for_continuous_domains.BAACOr()  
colony.set_verbosity(False)
colony.set_cost(flatten_cost(ackley))
colony.set_parameters(pop_size, k, min_q, max_q, min_xi, max_xi, 'lin', 'lin', [num_func_evals])
colony.define_variables(ranges, is_bounded)
solution = colony.optimize()
print(solution)

"""
# SA
print("SA")
# Parameters
initial_temperature = 10.0;  cooling_constant = 0.99;  step_size = 1e-2; 
local_iterations = 100
# Configure and run
sa = simulated_annealing.SA()
sa.set_verbosity(False)
sa.set_cost(flatten_cost(ackley))
sa.set_parameters(initial_temperature, cooling_constant, step_size, local_iterations, [num_func_evals])
sa.define_variables(ranges, is_bounded)
solution = sa.optimize()
print(solution)

# ACFSA
print("ACFSA")
# Parameters
initial_temperature = 10.0;  cooling_constant = 0.99;  step_size = 1e-2;
local_iterations = 100
# Configure and run
acfsa = simulated_annealing.ACFSA()
acfsa.set_verbosity(False)
acfsa.set_cost(flatten_cost(ackley))
acfsa.set_parameters(initial_temperature, cooling_constant, local_iterations, [num_func_evals])
acfsa.define_variables(ranges, is_bounded)
solution = acfsa.optimize()
print(solution)

# PSO
print("PSO")
# Parameters
swarm = particle_swarm_optimization.PSO()
swarm_size = 20;  personal_acceleration = 2;  global_acceleration = 2
# Configure and run
swarm.set_verbosity(False)
swarm.set_cost(flatten_cost(ackley))
swarm.set_parameters(swarm_size, personal_acceleration, global_acceleration, [num_func_evals])
swarm.define_variables(ranges, is_bounded)
solution = swarm.optimize()
print(solution)


# AIWPSO
print("AIWPSO")
# Parameters
swarm = particle_swarm_optimization.AIWPSO()
swarm_size = 20;  personal_acceleration = 2;  global_acceleration = 2
min_inertia = 0.3; max_inertia = 0.99
# Configure and run
swarm.set_verbosity(False)
swarm.set_cost(flatten_cost(ackley))
swarm.set_parameters(swarm_size, personal_acceleration, global_acceleration, min_inertia, max_inertia, [num_func_evals])
swarm.define_variables(ranges, is_bounded)
solution = swarm.optimize()
print(solution)
"""