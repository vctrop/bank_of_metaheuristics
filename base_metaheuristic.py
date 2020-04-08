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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    