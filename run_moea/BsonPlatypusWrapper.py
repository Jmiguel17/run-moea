import os
import uuid
import gzip
import random
import datetime

import numpy as np
import pandas as pd

from platypus import Archive
from bson.dbref import DBRef
from bson import Binary, Code
from bson.json_util import dumps
from pywr.optimisation.platypus import PlatypusWrapper
from pywr.optimisation import BaseOptimisationWrapper

import logging
logger = logging.getLogger(__name__)


class PyretoBasePlatypusWrapper(PlatypusWrapper):
    def __init__(self, *args, **kwargs):
        self.save_dataframes = kwargs.pop('save_dataframes', False)
        search_data = kwargs.pop('search_data', {})
        super().__init__(*args, **kwargs)

        search_data.update({
            'name': self.model.metadata['title'],
            'search_id': self.uid
        })
        search_data.update(**search_data)
        self._create_new_search_json(search_data)
        self.created_at = None

    def customise_model(self, model):
        model.solver.retry_solve = True

    def evaluate(self, solution):
        self.created_at = datetime.datetime.now()
        ret = super().evaluate(solution)
        self._create_new_individual_json()
        return ret

    def _create_new_search_json(self, search_data):

        if 'started_at' not in search_data:
            search_data['started_at'] = datetime.datetime.now().isoformat()
        self._save_new_search_json(search_data)

    def _save_new_search_json(self, search_data):
        raise NotImplementedError()

    def _create_new_individual_json(self):
        import time
        logger.info('Saving individual ...')
        t0 = time.time()

        evaluated_at = datetime.datetime.now()
        # TODO runtime statistics
        individual = dict(
            variables=list(self._generate_variable_documents()),
            metrics=list(self._generate_metric_documents()),
            created_at=self.created_at.isoformat(),
            evaluated_at=evaluated_at.isoformat(),
        )
        self._save_new_individual(individual)
        logger.info('Save complete in {:.2f}s'.format(time.time() - t0))

    def _save_new_individual(self, individual):
        raise NotImplementedError()

    def _generate_variable_documents(self):

        for variable in self.model.variables:

            if variable.double_size > 0:
                upper = variable.get_double_upper_bounds()
                lower = variable.get_double_lower_bounds()
                values = variable.get_double_variables()
                for i in range(variable.double_size):
                    yield dict(name='{}[d{}]'.format(variable.name, i), value=float(values[i]),
                                             upper_bounds=float(upper[i]), lower_bounds=float(lower[i]))

            if variable.integer_size > 0:
                upper = variable.get_integer_upper_bounds()
                lower = variable.get_integer_lower_bounds()
                values = variable.get_integer_variables()
                for i in range(variable.integer_size):
                    yield dict(name='{}[i{}]'.format(variable.name, i),
                               value=int(values[i]), upper_bounds=int(upper[i]), lower_bounds=int(lower[i]))

    def _generate_metric_documents(self):

        for recorder in self.model.recorders:

            try:
                value = float(recorder.aggregated_value())
            except NotImplementedError:
                value = None

            if self.save_dataframes:
                try:
                    df = recorder.to_dataframe()
                    df = Binary(df.to_msgpack())
                except AttributeError:
                    df = None
            else:
                df = None

            if value is not None or df is not None:
                yield dict(name=recorder.name, value=value,
                                       dataframe=df,
                                       objective=recorder.is_objective is not None,
                                       constraint=recorder.is_constraint,
                                       constraint_violated = recorder.is_constraint_violated() if recorder.is_constraint == True else False,
                                       minimise=recorder.is_objective == 'minimise')


class PyretoJSONPlatypusWrapper(PyretoBasePlatypusWrapper):
    def __init__(self, *args, **kwargs):
        self.output_directory = kwargs.pop('output_directory', 'outputs')
        super().__init__(*args, **kwargs)

    @property
    def output_subdirectory(self):
        path = os.path.join(self.output_directory, self.uid)
        os.makedirs(path, exist_ok=True)
        return path

    def _save_new_search_json(self, search_data):
        fn = os.path.join(self.output_subdirectory, 'search.bson')
        with open(fn, mode='w') as fh:
            fh.write(dumps(search_data))

    def _save_new_individual(self, individual):
        uid = uuid.uuid4().hex
        fn = os.path.join(self.output_subdirectory, 'i'+uid+'.bson.gz')

        with gzip.open(fn, mode='wt') as fh:
            fh.write(dumps(individual))


class LoggingArchive(Archive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.log = []
    
    def add(self, solution):
        super().add(solution)
        self.log.append(np.array([o for o in solution.objectives]))
