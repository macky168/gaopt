import gaopt
from gaopt import search_space

import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes

params_range={
    'lambda_l1': search_space.discrete_int(-8, 2),
    'lambda_l2': search_space.discrete_int(-8, 2),
    'num_leaves': search_space.discrete(2, 100, 4),
    'feature_fraction': search_space.discrete(0.1, 1.0, 0.02),
    'bagging_fraction': search_space.discrete(0.1, 1.0, 0.02),
    'bagging_freq': search_space.discrete_int(0,1),
    'min_child_samples': search_space.discrete_int(1,30),
}
cal_time_lst = []
date_start = None


def objective1(params):    
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.3, random_state = 0)
    X_train, X_valid, y_train, y_valid  = train_test_split(X_train, y_train, test_size = 0.3, random_state = 0)

    lgb_train = lgb.Dataset(data=X_train, label=y_train)
    lgb_valid = lgb.Dataset(data=X_valid, label=y_valid)
    
    params ={
        'lambda_l1': 10**params.lambda_l1,
        'lambda_l2': 10**params.lambda_l2,
        'num_leaves': params.num_leaves,
        'feature_fraction': params.feature_fraction,
        'bagging_fraction': params.bagging_fraction,
        'bagging_freq': params.bagging_freq,
        'min_child_samples': params.min_child_samples,
        'objective': 'regression',
        'metric': 'rmse',
        "verbosity": -1,
        "seed": 0
    }

    model = lgb.train(params,
                  train_set=lgb_train,
                  valid_sets=lgb_valid,
                  verbose_eval=False
                  )
    
    y_pred_lgb = model.predict(X_test)
    fitness = r2_score(y_test, y_pred_lgb)
    
    return fitness


def main():
    p_m = 0.10
    p_c = 0.7

    population = 30
    generation = 50

    instance = gaopt.GAOpt(params_range, objective=objective1, generation=generation, population=population,
                           p_m=p_m, p_c=p_c, elitism=True,
                           history=2, verbose=2, maximizing=True)
    best_params, best_fitness, best_fitness_lst, worst_fitness_lst, mean_fitness_lst, median_fitness_lst, sd_fitness_lst, search_history_lst = instance.fit()
    print("best params: ", best_params)
    print("best fitness: ", best_fitness)


if __name__ == '__main__':
    main()
