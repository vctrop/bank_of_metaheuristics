#!python3

# MIT License
# Copyright (c) 2020 Victor O. Costa
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
    
# Own
import ant_colony_for_continuous_domains
import particle_swarm_optimization
import simulated_annealing
import continuous_benchmarks
from deap.benchmarks import ackley

colony = ant_colony_for_continuous_domains.ACOr()
swarm = particle_swarm_optimization.PSO()
sa = simulated_annealing.SA()

num_func_evals = 1000

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
    
colony.set_verbosity(False)
colony.set_cost(continuous_benchmarks.ackley)
colony.set_parameters(num_iterations, pop_size, k, q, xi)
colony.define_variables(ranges, is_bounded)
solution = colony.optimize()
print(solution)

# SA
print("SA")
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

# PSO
print("PSO")
# Total # of function evaluations: population_size * (num_iterations + 1)
# Parameters to be used for PSO
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