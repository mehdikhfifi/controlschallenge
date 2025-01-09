
DATA_PATH = "/home/khfifi/controls_challenge/data"  
MODEL_PATH = "/home/khfifi/controls_challenge/models/tinyphysics.onnx"  

import multiprocessing
from pathlib import Path
import random
from deap import base, creator, tools
from tinyphysicsfile import run_rollout_with_pid  
import numpy as np


creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


toolbox.register("attr_float", random.uniform, -0.3, 0.3)


toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)









def evaluate_pid(individual):
    Kp, Ki, Kd = individual
    total_costs = []
    
    
    csv_files = list(Path(DATA_PATH).glob("*.csv"))[:10]
    
    
    for data_file in csv_files:
        cost_dict = run_rollout_with_pid(str(data_file), MODEL_PATH, p=Kp, i=Ki, d=Kd, debug=False)
        total_costs.append(cost_dict['total_cost'])
    
    
    avg_cost = np.mean(total_costs)
    
    
    return (-avg_cost,)

toolbox.register("evaluate", evaluate_pid)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga():
    population = toolbox.population(n=100)  
    NGEN = 200  
    for gen in range(NGEN):
        
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:  
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        
        for mutant in offspring:
            if random.random() < 0.2:  
                toolbox.mutate(mutant)
                del mutant.fitness.values

        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        best_ind = tools.selBest(population, 1)[0]
        print(f"Generation {gen}: Best PID = {best_ind}, Fitness = {best_ind.fitness.values[0]}")

    return tools.selBest(population, 1)[0]

if __name__ == "__main__":
    
    
    
    num_workers = multiprocessing.cpu_count() 
    pool = multiprocessing.Pool(processes=num_workers)
    
    
    toolbox.register("map", pool.map)

    best_pid = run_ga()
    print("Optimized PID parameters:", best_pid)

    pool.close()
    pool.join()
