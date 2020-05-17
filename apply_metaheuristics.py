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
     
   
# ACOr
# Total # of function evaluations: archive_size + population_size * num_iterations
print("ACOr")
k = 50;  pop_size = 10;  q = 0.01; xi = 0.85
num_iterations = (num_func_evals - k) / pop_size
if not (num_iterations.is_integer()):
    print("Error, number of function evaluations subtracted by k is not divisible by population size")
    exit(-1)
num_iterations = int(num_iterations)
print("# iterations = %d" % num_iterations)

colony = ant_colony_for_continuous_domains.ACOr()  
colony.set_verbosity(False)
colony.set_cost(ackley)
colony.set_parameters(num_iterations, pop_size, k, q, xi)
colony.define_variables(ranges, is_bounded)
solution = colony.optimize()
print(solution)

# ACSACOr
# Total # of function evaluations: archive_size + population_size * num_iterations
print("ACSACOr")
k = 50;  pop_size = 10; base_q = 0.01; xi = 0.85
min_q = base_q - base_q/2
max_q = base_q + base_q/2

num_iterations = (num_func_evals - k) / pop_size
if not (num_iterations.is_integer()):
    print("Error, number of function evaluations subtracted by k is not divisible by population size")
    exit(-1)
num_iterations = int(num_iterations)
print("# iterations = %d" % num_iterations)

colony = ant_colony_for_continuous_domains.ACSACOr()
colony.set_verbosity(False)
colony.set_cost(ackley)
colony.set_parameters(num_iterations, pop_size, k, xi, min_q, max_q, False)
colony.define_variables(ranges, is_bounded)
solution = colony.optimize()
print(solution)

# AGDACOr
# Total # of function evaluations: archive_size + population_size * num_iterations
print("AGDACOr")
k = 50;  pop_size = 10; base_xi = 0.85; q = 0.01
min_xi = base_xi - base_xi/2
max_xi = 0.98

num_iterations = (num_func_evals - k) / pop_size
if not (num_iterations.is_integer()):
    print("Error, number of function evaluations subtracted by k is not divisible by population size")
    exit(-1)
num_iterations = int(num_iterations)
print("# iterations = %d" % num_iterations)
  
colony = ant_colony_for_continuous_domains.AGDACOr()  
colony.set_verbosity(False)
colony.set_cost(ackley)
colony.set_parameters(num_iterations, pop_size, k, q, min_xi, max_xi, False)
colony.define_variables(ranges, is_bounded)
solution = colony.optimize()
print(solution)


# MAACOr
# Total # of function evaluations: archive_size + population_size * num_iterations
print("MAACOr")
k = 50;  pop_size = 10;  base_q = 0.01; base_xi = 0.85
min_xi = base_xi - base_xi/2
max_xi = 0.98
min_q = base_q - base_q/2
max_q = base_q + base_q/2
    
num_iterations = (num_func_evals - k) / pop_size
if not (num_iterations.is_integer()):
    print("Error, number of function evaluations subtracted by k is not divisible by population size")
    exit(-1)
num_iterations = int(num_iterations)
print("# iterations = %d" % num_iterations)
  
colony = ant_colony_for_continuous_domains.MAACOr()  
colony.set_verbosity(False)
colony.set_cost(ackley)
colony.set_parameters(num_iterations, pop_size, k, min_q, max_q, min_xi, max_xi, False, False)
colony.define_variables(ranges, is_bounded)
solution = colony.optimize()
print(solution)
"""
# SA
print("SA")
sa = simulated_annealing.SA()
# Total # of function evaluations: global_iter * local_iter + 1
# Parameters to be used for SA
initial_temperature = 10.0;  cooling_constant = 0.99;  step_size = 1e-2;
# Number of function evaluations for SA: global_iterations * local_iterations
local_iterations = 100
global_iterations = num_func_evals / local_iterations
print("# local/global iterations = %d/%d" % (local_iterations, global_iterations)) 
if not (global_iterations.is_integer()):
    print("Error, number of function evaluations is not divisible by number of local iterations")
    exit(-1)
global_iterations = int(global_iterations)
sa.set_verbosity(False)
sa.set_cost(continuous_benchmarks.ackley)
sa.set_parameters(global_iterations, local_iterations, initial_temperature, cooling_constant, step_size)
sa.define_variables(ranges, is_bounded)
solution = sa.optimize()
print(solution)

# ACFSA
print("ACFSA")
acfsa = simulated_annealing.ACFSA()
# Total # of function evaluations: global_iter * local_iter + 1
# Parameters to be used for ACFSA
initial_temperature = 10.0;  cooling_constant = 0.99;  step_size = 1e-2;
# Number of function evaluations for ACFSA: global_iterations * local_iterations
local_iterations = 100
global_iterations = num_func_evals / local_iterations
print("# local/global iterations = %d/%d" % (local_iterations, global_iterations)) 
if not (global_iterations.is_integer()):
    print("Error, number of function evaluations is not divisible by number of local iterations")
    exit(-1)
global_iterations = int(global_iterations)
acfsa.set_verbosity(False)
acfsa.set_cost(continuous_benchmarks.ackley)
acfsa.set_parameters(global_iterations, local_iterations, initial_temperature, cooling_constant)
acfsa.define_variables(ranges, is_bounded)
solution = acfsa.optimize()
print(solution)

# PSO
print("PSO")
# Total # of function evaluations: population_size * (num_iterations + 1)
# Parameters to be used for PSO
swarm = particle_swarm_optimization.PSO()
swarm_size = 20;  personal_acceleration = 2;  global_acceleration = 2
# Number of function evaluations for PSO: swarm_size * num_iterations
num_iterations = num_func_evals / swarm_size
print("# iterations = %d" % num_iterations) 
if not (num_iterations.is_integer()):
    print("Error, number of function evaluations is not divisible by swarm size")
    exit(-1)
num_iterations = int(num_iterations)
swarm.set_verbosity(False)
swarm.set_cost(continuous_benchmarks.ackley)
swarm.set_parameters(100, 55, 2, 2)
swarm.define_variables(ranges, is_bounded)
solution = swarm.optimize()
print(solution)
"""