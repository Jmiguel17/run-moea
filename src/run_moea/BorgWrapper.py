import os
import uuid
import gzip
import random
import datetime

import numpy as np
import pandas as pd
from .borg import *

from platypus import Archive
from bson.dbref import DBRef
from bson import Binary, Code
from bson.json_util import dumps
from pywr.optimisation import cache_objectives, cache_variable_parameters, BaseOptimisationWrapper

import logging
logger = logging.getLogger(__name__)


class BorgWrapper(BaseOptimisationWrapper):
    """ A helper class for running pywr optimisations with borg.
    """
    def __init__(self, filename, seed, max_nfe, use_mpi, frequency, *args, **kwargs):
        super().__init__(filename)
        self.filename = filename

        self.seed = seed
        self.max_nfe = max_nfe
        self.use_mpi = use_mpi
        self.frequency = frequency

        directory, model_name = os.path.split(self.filename)
        output_directory = os.path.join(directory, 'outputs')
        
        output_directory = os.path.join(output_directory, 'outputs')

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        if self.seed is None:
            self.seed = random.randrange(sys.maxsize)

        if self.seed is not None:
            random.seed(self.seed)

        runtime_file = f'{model_name[0:-5]}_seed-{seed}_runtime%d.txt'
        self.runtime_file_path = os.path.join(output_directory, runtime_file)


    def _make_variables(self, variables):
        """Setup the variable types. """

        variablesbounds = []
        for var in variables:
            if var.double_size > 0:
                lower = var.get_double_lower_bounds()
                upper = var.get_double_upper_bounds()
                for i in range(var.double_size):
                    tem = [lower[i], upper[i]]
                    variablesbounds.append(tem)

            if var.integer_size > 0:
                lower = var.get_integer_lower_bounds()
                upper = var.get_integer_upper_bounds()
                for i in range(var.integer_size):
                    # Integers are cast to real
                    tem = [lower[i], upper[i]]
                    variablesbounds.append(tem)

        return variablesbounds
    
    
    def evaluate(self, solution):
        logger.info('Evaluating solution ...')

        for ivar, var in enumerate(self.model_variables):
            j = slice(self.model_variable_map[ivar], self.model_variable_map[ivar+1])
            x = np.array(solution[j])
            assert len(x) == var.double_size + var.integer_size
            if var.double_size > 0:
                var.set_double_variables(np.array(x[:var.double_size]))

            if var.integer_size > 0:
                ints = np.round(np.array(x[-var.integer_size:])).astype(np.int32)
                var.set_integer_variables(ints)

        run_stats = self.model.run()

        objectives = []
        for r in self.model_objectives:
            sign = 1.0 if r.is_objective == 'minimise' else -1.0
            value = r.aggregated_value()
            objectives.append(sign*value)

        # Return values to the solution
        logger.info('Evaluation complete!')

        return objectives
    
    def run(self):
        """ Run the optimisation. """

        # To determine the number of variables, etc
        m = self.model

        # Cache the variables, objectives and constraints
        variables, variable_map = cache_variable_parameters(m)
        objectives = cache_objectives(m)

        if len(variables) < 1:
            raise ValueError('At least one variable must be defined.')

        if len(objectives) < 1:
            raise ValueError('At least one objective must be defined.')

        # Accessing to the epsilon property
        epsilons = []
        for obj in objectives:
            tem = obj.epsilon
            epsilons.append(tem)

        problem = Borg(variable_map[-1], len(objectives), 0, self.evaluate)
        problem.setEpsilons(*epsilons)
        problem.setBounds(*self._make_variables(variables))

        if self.use_mpi:
            print(f"Running Borg with {self.max_nfe} evaluations and mpi {self.use_mpi}")
            Configuration.startMPI()
            results = problem.solveMPI(islands=1, maxEvaluations=self.max_nfe, runtime=self.runtime_file_path)
            
            if results is not None:
                print(results)
            
            logger.info('Optimisation complete')

            Configuration.stopMPI()
        
        else:
            print(f"Running Borg with {self.max_nfe} evaluations")
            results = problem.solve({"maxEvaluations": self.max_nfe, "runtimeformat": 'borg', "frequency": self.frequency,
                                    "runtimefile": self.runtime_file_path})
            
            if results is not None:
                print(results)
            
            logger.info('Optimisation complete')
