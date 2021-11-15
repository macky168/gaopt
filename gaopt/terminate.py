def terminate(best_fitness_lst, mean_fitness_lst):
    # ---------------------
    # There is no improve of best_fitness and mean_fitness, we will terminate.
    # ---------------------
    if len(best_fitness_lst) <= 1:
        return False
    elif best_fitness_lst[-1] <= best_fitness_lst[-2] and mean_fitness_lst[-1] <= mean_fitness_lst[-2]:
        return True
    else:
        return False

# reference:
# https://stackoverflow.com/questions/8462678/termination-conditions-for-genetic-algorithm
