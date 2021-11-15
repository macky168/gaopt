import copy
import random
import datetime

import numpy as np
import tensorflow as tf
import pandas as pd

from .mutate import mutate_normal
from .terminate import terminate
from .get_info_about_params import get_max_params, \
    get_max_fitness, get_min_fitness, get_mean_fitness, get_median_fitness, get_sd_fitness
from .selectparent import SelectParent

"""
Hyper-parameters need to be explained:
    - params: search_range (explained below)
    - objective: objective function
    - generation [int]: generation size (default: 30)
    - population [int]: population size (population: 100)
    - p_m [float]: mutation rate (default: 0.1=10%)
    - p_c [float]: crossover rate (default: 0.7=70%)
    - elitism: use elitism or not (default: True) 
    - rate_of_roulette_wheel_selection_at_crossover: roulette wheel selection or ranking selection (default: 0.0 =all ranking selection), 
    - rate_of_roulette_wheel_selection_at_copy_parent: roulette wheel selection or ranking selection (default: 0.0 =all ranking selection),
    - early_stopping [True or False]: whether you use early_stopping or not (default: False)
        - if no improve of best_fitness and mean_fitness, terminate.
    - maximizing [True or False]: maximizing or minimizing problem 
    - preserving_calc [True or False]: preserving calculation or not
    - history [2, 1, or 0]: outputs the detail or less (default: 0)
        - 2: all information;   best_params, best_fitness, 
                                best_fitness_lst, worst_fitness_lst, 
                                mean_fitness_lst, median_fitness_lst, sd_fitness_lst
        - 1: best_params, best_fitness
        - 0: best_params only
    - verbose [2, 1, or 0]: print the detail or less at each step (default: 0)
        - 2: detail
        - 1: less
        - 0: nothing
    - seed [int]: seed at randomizing

Params range should be specified as follows.

    from gaopt import search_space
    params = {
        'x1': search_space.categorical(['relu', 'tanh']), # list(candidates)
        'x2': search_space.discrete(-1.0, 1.0, 0.2), # min, max, step
        'x3': search_space.discrete_int(-4, 2), # min, max
        'x4': search_space.fixed(1) # a fixed value
    }



Hiroya MAKINO
Grad. School of Informatics, Nagoya University

ver1.0, Apr. 30 2021
"""


class params_comb:
    pass


class GAOpt:
    
    def __init__(
            self, 
            params, objective, generation=50, population=100,
            p_m=0.10, p_c=0.7, elitism=True, rate_of_roulette_wheel_selection_at_crossover=0.0, rate_of_roulette_wheel_selection_at_copy_parent=0.0,
            early_stopping=False, maximizing=True, preserving_calc=True, 
            history=0, verbose=0, seed=168):
        if params is None:
            TypeError("You must specify the params range")
        self.params = params
        self.keys = [key for key in params.keys()]
        for key in self.keys:
            setattr(params_comb, key, "")
            
        if objective is None:
            TypeError("You must specify the objective function")
        self.objective = objective
        
        self.num_of_gens = generation
        if population % 2 == 1:
            TypeError("Population must be set in even number")
        self.population = population
        
        self.rate_of_mutation = p_m
        self.rate_of_crossover = p_c
        self.elitism = elitism
        self.rate_of_roulette_wheel_selection_at_crossover = rate_of_roulette_wheel_selection_at_crossover
        self.rate_of_roulette_wheel_selection_at_copy_parent = rate_of_roulette_wheel_selection_at_copy_parent
                
        self.early_stopping = early_stopping
        self.maximizing = maximizing
        self.preserving_calc = preserving_calc
        self.history = history
        self.verbose = verbose

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
    def fit(self):
        current_lst = [[] for pop in range(self.population)]

        # ---------------------
        # initial population
        # ---------------------
        for i in range(self.population):
            temp_params_comb = params_comb()
            for key in self.keys:
                setattr(temp_params_comb, key, self.params[key].select())
            current_lst[i] = temp_params_comb
        next_lst = copy.deepcopy(current_lst)
        
        best_params_lst = []
        best_fitness_lst = []
        worst_fitness_lst = []
        mean_fitness_lst = []
        median_fitness_lst = []
        sd_fitness_lst = []
        
        calc_fitnesses_lst = []
        calc_params_combs = []

        for gen in range(self.num_of_gens):
            print('\n')
            print('*** generation', gen, "/", self.num_of_gens, " *************")

            current_lst = copy.deepcopy(next_lst)
            current_fitness_lst = [None] * self.population
            next_lst = [0] * self.population

            # ---------------------
            # maximizing
            # ---------------------
            if self.maximizing:
                # ---------------------
                # calculate fitness
                # ---------------------
                for i in range(self.population):
                    if self.verbose > 0:
                        print("\r{0} {1} {2} {3} ".
                              format('    calculating', i + 1, '/', self.population), end="")
                        
                    if self.preserving_calc:
                        if len(calc_params_combs) > 0:
                            for calc_params_comb_index in range(len(calc_params_combs)):
                                if self.is_params_comb_same(calc_params_combs[calc_params_comb_index], 
                                                            current_lst[i]):
                                    current_fitness_lst[i] = calc_fitnesses_lst[calc_params_comb_index]
                                    break
                                elif calc_params_comb_index == len(calc_params_combs) -1:
                                    score = self.output_fitness(current_lst[i])
                                    current_fitness_lst[i] = score
                                    calc_fitnesses_lst.append(score)
                                    calc_params_combs.append(copy.deepcopy(current_lst[i]))
                        else:   
                            score = self.output_fitness(current_lst[0])
                            current_fitness_lst[0] = score
                            calc_fitnesses_lst.append(score)
                            calc_params_combs.append(copy.deepcopy(current_lst[0]))
                    else:
                        current_fitness_lst[i] = self.output_fitness(current_lst[i])

                # ---------------------
                # information
                # ---------------------
                best_params_lst += [get_max_params(current_lst, current_fitness_lst, self.keys)]
                best_fitness_lst += [get_max_fitness(current_fitness_lst)]
                worst_fitness_lst += [get_min_fitness(current_fitness_lst)]
                mean_fitness_lst += [get_mean_fitness(current_fitness_lst)]
                median_fitness_lst += [get_median_fitness(current_fitness_lst)]
                sd_fitness_lst += [get_sd_fitness(current_fitness_lst)]

                if self.verbose > 1:
                    print('\n')
                    print('best parameters is : ', end="")
                    v_index = 0
                    for k in self.keys:
                        print(k, "", best_params_lst[-1][v_index], end=",   ")
                        v_index += 1
                    print('')
                    print('best fitness is : ', best_fitness_lst[-1])
                    print('worst fitness is : ', worst_fitness_lst[-1])
                    print('mean fitness is : ', mean_fitness_lst[-1])
                    print('median fitness is : ', median_fitness_lst[-1])
                    print('sd fitness is : ', sd_fitness_lst[-1])
                    print('\n')

            # ---------------------
            # minimizing
            # ---------------------
            else:
                # ---------------------
                # calculate fitness
                # ---------------------
                for i in range(self.population):
                    if self.verbose > 0:
                        print("\r{0} {1} {2} {3} ".
                              format('    calculating', i + 1, '/', self.population), end="")
                        
                    if self.preserving_calc:
                        if len(calc_params_combs) > 0:
                            for calc_params_comb_index in range(len(calc_params_combs)):
                                if self.is_params_comb_same(calc_params_combs[calc_params_comb_index], 
                                                            current_lst[i]):
                                    current_fitness_lst[i] = calc_fitnesses_lst[calc_params_comb_index]
                                    break
                                elif calc_params_comb_index == len(calc_params_combs) -1:
                                    score = -self.output_fitness(current_lst[i])
                                    current_fitness_lst[i] = score
                                    calc_fitnesses_lst.append(score)
                                    calc_params_combs.append(copy.deepcopy(current_lst[i]))
                        else:   
                            score = -self.output_fitness(current_lst[0])
                            current_fitness_lst[0] = score
                            calc_fitnesses_lst.append(score)
                            calc_params_combs.append(copy.deepcopy(current_lst[0]))
                    else:
                        current_fitness_lst[i] = -self.output_fitness(current_lst[i])
                        
                # ---------------------
                # information
                # ---------------------
                best_params_lst += [get_max_params(current_lst, current_fitness_lst, self.keys)]
                best_fitness_lst += [-get_max_fitness(current_fitness_lst)]
                worst_fitness_lst += [-get_min_fitness(current_fitness_lst)]
                mean_fitness_lst += [-get_mean_fitness(current_fitness_lst)]
                median_fitness_lst += [-get_median_fitness(current_fitness_lst)]
                sd_fitness_lst += [get_sd_fitness(current_fitness_lst)]

                if self.verbose > 1:
                    print('\n')
                    print('best parameters is : ', end="")
                    v_index = 0
                    for k in self.keys:
                        print(k, "", best_params_lst[-1][v_index], end=",   ")
                        v_index += 1
                    print('')
                    print('best fitness is : ', best_fitness_lst[-1])
                    print('worst fitness is : ', worst_fitness_lst[-1])
                    print('mean fitness is : ', mean_fitness_lst[-1])
                    print('median fitness is : ', median_fitness_lst[-1])
                    print('sd fitness is : ', sd_fitness_lst[-1])
                    print('\n')

            # ---------------------
            # termination
            # ---------------------
            if self.early_stopping:
                if terminate(best_fitness_lst, mean_fitness_lst):
                    break
                
            # ---------------------
            # crossover / copy
            # ---------------------
            j = 0
            while j < self.population:
                key1 = random.random()

                # ---------------------
                # crossover
                # ---------------------
                if key1 < self.rate_of_crossover and j+1 < self.population:
                    next_lst[j], next_lst[j+1] \
                        = copy.deepcopy(self.crossover(current_lst, current_fitness_lst))
                    j += 2

                # ---------------------
                # copy
                # ---------------------
                else:
                    next_lst[j], next_lst[j+1] \
                        = copy.deepcopy(self.copy_parent(current_lst, current_fitness_lst))
                    j += 2
                        
            # ---------------------
            # mutation
            # ---------------------
            for i in range(self.population):
                next_lst[i] = copy.deepcopy(
                    mutate_normal(next_lst[i], self.params, self.keys, self.rate_of_mutation))
        
            # ---------------------
            # elitism: preserve the best individual
            # ---------------------
            if self.maximizing:
                best_index = current_fitness_lst.index(max(current_fitness_lst))
            else:
                best_index = current_fitness_lst.index(min(current_fitness_lst))
            
            if self.elitism:
                for k in range(self.population):
                    if self.is_params_comb_same(current_lst[best_index], next_lst[k]):
                        break
                    if k == self.population - 1:
                        next_lst[k] = copy.deepcopy(current_lst[best_index])

                
        if self.maximizing:
            best_index = best_fitness_lst.index(max(best_fitness_lst))
        else:
            best_index = best_fitness_lst.index(min(best_fitness_lst))
            
        if self.history == 2:
            return best_params_lst[best_index], best_fitness_lst[best_index],\
                   best_fitness_lst, worst_fitness_lst, mean_fitness_lst, median_fitness_lst, sd_fitness_lst
        elif self.history == 1:
            return best_params_lst[best_index], best_fitness_lst[best_index]
        elif self.history == 0:
            return best_params_lst[best_index]

    def output_fitness(self, params_comb_temp):
        fitness = self.objective(params_comb_temp)
        return fitness
            
    def is_params_comb_same(self, a, b):
        result = True
      
        for key in self.keys:
            if getattr(a, key) != getattr(b, key):
                result = False
                break
        return result
    
    def crossover(self, current_lst, current_fitness_lst):
        chromosome_length = len(self.keys)
        
        key1 = random.random()
        if key1 < self.rate_of_roulette_wheel_selection_at_crossover:
            instance_roulette = SelectParent(current_lst, current_fitness_lst)
            parent1 = instance_roulette.roulette_select()
            parent2 = instance_roulette.roulette_select()
        else:           # ranking selection
            instance_ranking = SelectParent(current_lst, current_fitness_lst)
            parent1 = instance_ranking.ranking_select()
            parent2 = instance_ranking.ranking_select()

        child1 = params_comb()
        child2 = params_comb()

        point_a = random.randint(0, chromosome_length-1)
        point_b = random.randint(0, chromosome_length-1)
        point1 = min(point_a, point_b)
        point2 = max(point_a, point_b)
        
        i = 0
        while i < chromosome_length:
            if i <= point1 or i > point2:
                setattr(child1, self.keys[i], getattr(parent1, self.keys[i]))
                setattr(child2, self.keys[i], getattr(parent2, self.keys[i]))
            else:
                setattr(child1, self.keys[i], getattr(parent2, self.keys[i]))
                setattr(child2, self.keys[i], getattr(parent1, self.keys[i]))
            i += 1
        
        return child1, child2
    
    def copy_parent(self, current_lst, current_fitness_lst):
        key1 = random.random()
        if key1 < self.rate_of_roulette_wheel_selection_at_copy_parent:
            instance_roulette = SelectParent(current_lst, current_fitness_lst)
            child1 = instance_roulette.roulette_select()
            child2 = instance_roulette.roulette_select()
            
        else:           # ranking selection
            instance_ranking = SelectParent(current_lst, current_fitness_lst)
            child1 = instance_ranking.ranking_select()
            child2 = instance_ranking.ranking_select()

        return child1, child2
