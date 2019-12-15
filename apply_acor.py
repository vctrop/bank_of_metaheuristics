# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 18:16:01 2018

@author: Victor Costa

Simple example of how to apply the ACOr class.
"""

import ant_colony_for_continuous_domains
import continuous_benchmarks

colony = ant_colony_for_continuous_domains.ACOr()
ranges = [[-5,5],
          [-5,5],
          [-5,5]]

colony.set_cost(continuous_benchmarks.sphere)
colony.set_parameters(100, 5, 50, 0.01, 0.85)
colony.set_variables(3, ranges)

solution = colony.optimize()

print(solution)