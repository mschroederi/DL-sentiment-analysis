import itertools as it
import math
from torch import nn

from typing import Callable

class GridSearch:
    def __init__(self, param_grid: dict, run_with_config: Callable[[dict], float]):
        super().__init__()
        self.param_grid = param_grid
        self.run_with_config = run_with_config

        self.best_config = None
        self.best_model = None
        self.best_valid_loss = math.inf

        self.all_results = []
    
    def __get_configuration(self, config_tuple):
        config = dict()
        for i, k in enumerate(self.param_grid.keys()):
            config[k] = config_tuple[i]
        return config

    def __get_configurations(self):
        configs = it.product(*(self.param_grid[k] for k in self.param_grid))
        configs = [self.__get_configuration(c) for c in list(configs)]
        return configs

    def run(self, num_epochs: int = 50, patience: int = 5):
        configs = self.__get_configurations()
        print('Found {} different configurations'.format(len(configs)))
        print('Starting grid search now...')
        for i, config in enumerate(configs):
            print('\nRunning {} of {} with configuration {}'.format(i+1, len(configs), config))
            l, model = self.run_with_config(config, num_epochs, patience)
            self.all_results.append({'config': config, 'valid_loss': l})
            if l < self.best_valid_loss:
                print('\nGridSearch: Updating best model. Validation loss decreased from {} to {}'.format(self.best_valid_loss, l))
                self.best_valid_loss = l
                self.best_config = config
                self.best_model = model
        print('\nCompleted grid search')
        print('Best validation loss of {} was reached with configuration {}'.format(self.best_valid_loss, self.best_config))
        return self.best_model
