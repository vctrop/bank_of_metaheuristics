# Bank of metaheuristics

Accurate and comprehensible implementation of multiple metaheuristics.

# Third party software versions:
* Python 3.6.9
  * NumPy 1.17.3 (vector math)
  * Deap 1.3 (only to import benchmark functions)

## Metaheuristics implemented up to now:
<pre>
- Ant colony optimization for continuous domains (ACOr).    Socha, 2006.
- Adaptive elitism level ACOr (AELACOr).                    Costa, 2020.
- Adaptive generation dispersion ACOr (AGDACOr).            Costa, 2020.
- Bi-adaptive ACOr (MAACOr).                                Costa, 2020.
- Simulated annealing (SA).                                 Kirkpatrick, 1983.
- Adaptive crystallization factor SA (ACFSA).               Martins, 2012.
- Particle swarm optimization (PSO).                        Kennedy, 1995.
- Adaptive inertia weight PSO (AIWPSO).                     Nickabadi, 2011.
</pre>

## List of modules
* base_metaheuristic.py
    + simulated_annealing.py
    + particle_swarm_optimization.py
    + ant_colony_for_continuous_domains.py

## Scripts and their uses
* apply_metaheuristics.py   - Uses all metaheuristics to search for minimum values of a given benchmark (mostly used for verification purposes)
* lin_sig_exp_experiment.py - Extracts results for AELACOr and AGDACOr considering different maps from the colony success rate to parameter values
* lin_sig_exp_stats.py      - Displays summary statistics for the results from lin_sig_exp_experiment.py
* metaheuristic_test_functions_experiment.py   - Collects results for a given metaheuristics in a set of test function instances.
* metaheuristic_results_tables.py              - Displays summary statistics and statistical significance of the results from metaheuristic_test_functions_experiment.py
* metaheuristic_results_plot.py                - Plots the average search history of each metaheuristic, considering results from metaheuristic_test_functions_experiment.py

### If this repository is valuable to you, consider citing:
Costa, V. O. and Müller, M. F. (2020). "On the Multiple Possible Adaptive Mechanisms of the Continuous Ant Colony Optimization". <i>  9th Brazilian Conference on Intelligent Systems, BRACIS </i> (2020).
