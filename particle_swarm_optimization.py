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
# Own
from base_metaheuristic import Base
    
class PSO(Base):
    """ Class for the Particle Swarm Optimization algorithm (PSO), following (Poli et al., 2007) """

    def __init__(self):
        """ Constructor """
        # Define verbosity and NULL problem definition
        super().__init__
        
        # Initial algorithm parameters
        self.relative_iterations = None         # Array containing the iterations at which best solutions are reported
        self.num_iter = 0                       # Total number of iterations
        self.population_size = 0                # Number of particles
        self.personal_acceleration = 0.5        # Tendency towards personal bests
        self.global_acceleration = 0.5          # Tendency towards global best
        self.inertia_weight = 1.0               # Inertia weight constant at one is the same as no inertia weight
        
        # Optimization results
        self.swarm_positions = None             # Current solutions of the swarm
        self.swarm_velocities = None            # Current velocities (perturbations) of each particle in the swarm
        self.personal_bests = None              # Best solution each particle encountere during the search
        self.global_best = None                 # Best solution found in the search
        
        # Flag for modified PSO
        self.adaptive_inertia = False           # In vanilla PSO, there is no inertia weighting
        
        
    def set_parameters(self, population_size, personal_acceleration, global_acceleration, function_evaluations_array):
        """ Define values for the parameters used by the algorithm """
        # Input error checking
        if population_size <= 0:
            print("Population size must be greater than zero")
            exit(-1)
        if personal_acceleration < 0 or global_acceleration < 0:
            print("Personal and global weights must be equal or greater than zero")
            exit(-1)
        if len(function_evaluations_array) == 0:
            print("Error, objective function evaluation array must not be empty")
            exit(-1)
            
        # Number of function evaluations for PSO: population_size * num_iterations 
        function_evaluations_array = np.array(function_evaluations_array)
        self.relative_iterations = function_evaluations_array / population_size
        all_divisible = (np.array([x.is_integer() for x in self.relative_iterations])).all()
        if not all_divisible:
            print("Error, at least one number of function evaluations is not divisible by population size")
            exit(-1)
        
        self.num_iter = int(np.max(self.relative_iterations))
        self.population_size = population_size
        self.personal_acceleration = personal_acceleration
        self.global_acceleration = global_acceleration
        
    
    def define_variables(self, initial_ranges, is_bounded):
        """ Defines the number of variables, their initial values ranges and wether or not these ranges constrain the variable during the search """
        # Input error checking
        if self.num_iter == 0:
            print("Error, please set algorithm parameters before variables definition")
            exit(-1)
        if len(initial_ranges) == 0 or len(is_bounded) == 0:
            print("Error, initial_ranges and is_bounded lists must not be empty")
            exit(-1)
        if len(initial_ranges) != len(is_bounded):
            print("Error, the number of variables for initial_ranges and is_bounded must be equal")
            exit(-1)
        
        self.num_variables = len(initial_ranges)
        self.initial_ranges = initial_ranges
        self.is_bounded = is_bounded
        
        self.swarm_positions = np.zeros((self.population_size, self.num_variables + 1))
        self.swarm_velocities = np.zeros((self.population_size, self.num_variables))
        
        # Personal and global bests initially have infinite cost
        self.personal_bests = np.zeros((self.population_size, self.num_variables + 1))
        self.personal_bests[:, -1] = float('inf')
        self.global_best = np.zeros(self.num_variables + 1)
        self.global_best[-1] = float('inf')
            

    def update_inertia_weight(self, acceptance_count):
        """ Inertia weight is not updated in vanilla PSO. It is kept at 1.0, the same of determining no inertia weight """
        pass
            
            
    def optimize(self):
        """ Initializes the archive and enter the main loop, until it reaches maximum number of iterations """
        # Variables and cost function must be defined prior to optimization
        if self.num_variables == None:
            print("Error, number of variables and their boundaries must be defined prior to optimization")
            exit(-1)
        if self.cost_function == None:
            print("Error, cost function must be defined prior to optimization")
            exit(-1)
        
        # Initialize swarm positions and velocities randomly
        for i in range(self.population_size):
            for j in range(self.num_variables):
                self.swarm_positions[i, j] = np.random.uniform(self.initial_ranges[j][0], self.initial_ranges[j][1])
                self.swarm_velocities[i, j] = np.random.uniform(self.initial_ranges[j][0], self.initial_ranges[j][1])
        
        # Keep solutions defined by function_evaluations_array
        recorded_solutions = []
        
        # Main optimization loop (population_size * num_iter cost function evaluations)
        for iteration in range(self.num_iter):
            # When using adaptive inertia weight
            acceptance_count = 0
            
            for particle in range(self.population_size):
                # Compute cost of new position
                self.swarm_positions[particle, -1] = self.cost_function(self.swarm_positions[particle, :-1])
                
                # Update personal best solution
                if self.swarm_positions[particle, -1] < self.personal_bests[particle, -1]:
                    self.personal_bests[particle, :] = np.array(self.swarm_positions[particle, :])
                    acceptance_count += 1
                    
                    # Update global best solution
                    if self.personal_bests[particle, -1] < self.global_best[-1]:
                        self.global_best = np.array(self.personal_bests[particle, :])
                        
                # Update inertia weight based on success rate of the swarm
                # Has no effect in vanilla PSO
                self.update_inertia_weight(acceptance_count)
                
                # Update velocity vector
                self.swarm_velocities[particle, :] =    self.inertia_weight * (self.swarm_velocities[particle, :]
                                                        + self.personal_acceleration  * np.random.rand() * (self.personal_bests[particle, :-1]    - self.swarm_positions[particle, :-1])
                                                        + self.global_acceleration    * np.random.rand() * (self.global_best[:-1]                 - self.swarm_positions[particle, :-1]))
                # Update position vector
                self.swarm_positions[particle, :-1] = self.swarm_positions[particle, :-1] + self.swarm_velocities[particle, :]
                
                # Restrict search for bounded variables
                for var in range(self.num_variables):
                    if self.is_bounded[var]:
                        # Use the hard border strategy
                        if self.swarm_positions[particle, var] < self.initial_ranges[var][0]:
                            self.swarm_positions[particle, var] = self.initial_ranges[var][0]
                        elif self.swarm_positions[particle, var] > self.initial_ranges[var][1]:
                            self.swarm_positions[particle, var] = self.initial_ranges[var][1]        
            
            if (self.relative_iterations - 1 == iteration).any():
                recorded_solutions.append(np.array(self.global_best))
            
        return np.array(recorded_solutions)
        
        
class AIWPSO(PSO):
    """ Class for the Adaptative Inertia Weight Particle Swarm Optimization (AIWPSO), following (Nickabadi et al., 2011).
        Only the adaptive mechanism of AIWPSO is implemented here.
        The paper also uses a mutation mechanism for the worst particle at each iteration, which is left unimplemented. """
        
    def __init__(self):
        """ Constructor """
        super().__init__()
        
        self.adaptive_inertia = True
        self.max_inertia = None
        self.min_inertia = None
    
    def set_parameters(self, population_size, personal_acceleration, global_acceleration, min_inertia, max_inertia, function_evaluations_array):
        if min_inertia > max_inertia:
            print("Max intertia mut be greater than min inertia")
            exit(-1)
            
        super().set_parameters(population_size, personal_acceleration, global_acceleration, function_evaluations_array)
        self.min_inertia = min_inertia
        self.max_inertia = max_inertia
        
    def update_inertia_weight(self, acceptance_count):
        """ Use swarm success rate to update the inertia weight """
        success_percentage = acceptance_count / self.population_size
        self.inertia_weight = (self.max_inertia - self.min_inertia) * success_percentage + self.min_inertia
        
        
        
        