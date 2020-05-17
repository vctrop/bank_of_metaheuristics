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

from abc import ABC, abstractmethod

class Base:
    """ """
    
    def __init__(self):
        """ Constructor """
        self.verbosity = True
        
        # Initial (NULL) problem definition
        self.num_variables = None                       # Number of variables
        self.initial_ranges = []                        # Initialization boundaries for each variable
        self.is_bounded = []                            # Here, if a variable is constrained, it will be limited to its initialization boundaries for all the search
        self.cost_function = None                       # Cost function to guide the search
        
    
    def set_verbosity(self, status):
        """ If verbosity is set True, print partial results of the search will be printed """
        # Input error checking
        if not (type(status) is bool):
            print("Error, verbosity parameter must be a boolean")
            exit(-1)
            
        self.verbosity = status
    
    
    def set_cost(self, cost_function):
        """ Sets the cost function that will guide the search """
        self.cost_function = cost_function
    
    @abstractmethod
    def define_variables(self, initial_ranges, is_bounded):
        pass
    
    @abstractmethod
    def optimize(self):
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    