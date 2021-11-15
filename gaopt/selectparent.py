import numpy as np


class SelectParent:
    def __init__(self, previous_lst, previous_fitness_lst):
        self.previous_lst = previous_lst
        self.previous_fitness_lst = [round(float(i), 5) for i in previous_fitness_lst]
        self.index_lst = [i for i in range(len(previous_lst))]

    # ---------------------
    # roulette wheel selection
    # ---------------------
    def roulette_select(self):
        prob = np.array(self.previous_fitness_lst) / sum(self.previous_fitness_lst)
        prob = [round(prob[i], 5) for i in range(len(prob))]
        prob[0] = 1.00000 - round(sum(prob[1:]), 5)
        # prob.sum must be equal to 1
        # This sentence is needed because we have rounding error.

        parent1_index = np.random.choice(self.index_lst, p=prob)
        parent1 = self.previous_lst[parent1_index]

        return parent1

    # ---------------------
    # ranking selection
    # ---------------------
    def ranking_select(self):
        temp_lst = np.concatenate((np.array(self.index_lst).reshape(-1, 1),
                                   np.array(self.previous_fitness_lst).reshape(-1, 1)), axis=1)
        sorted_index_and_previous = sorted(temp_lst, key=lambda x: float(x[1]), reverse=True)
        sorted_index = [0]*len(self.previous_lst)

        for j in range(len(sorted_index_and_previous)):
            sorted_index[j] = sorted_index_and_previous[j][0].astype(int)

        prob = [0]*len(sorted_index)
        for k in range(len(sorted_index)):
            prob[sorted_index[-k-1]] = k + 1
        prob = np.array(prob).astype(float) / sum(prob)
        prob = [round(prob[i], 5) for i in range(len(prob))]
        prob[0] = 1.00000 - round(sum(prob[1:]), 5)

        parent1_index = np.random.choice(self.index_lst, p=prob)
        parent1 = self.previous_lst[parent1_index]

        return parent1
