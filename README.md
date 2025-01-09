report:

pid: modified the base PID controller to accept dynamic parameters (p,i,d) during initialization instead of hard coded values
rollout function: made new rollout function that runs simulation rollout with given pid values, returning total cost for fitness 
genetic algorithm: i used the deap library, defining individual representations, population initalization and standard ga operations (selection, crossover…)
fitness evaluation: implemented evaluation_pid function that instantiates the pid with candidate parameters, runs simulation on one or more data segment,a nd returns a fitness score based on negative total cost
multiprocessing: i used python's multiprocessing package to quickly find suitable pid parameters, and increase cpu utilization

once parameters were found they were hardcoded into w_pid.
i found that evaluation on a single csv was better than on multiple, and did not overfit
