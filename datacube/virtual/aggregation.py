import xarray
import numpy as np
import pandas as pd

from collections import Sequence
from functools import partial
from .impl import VirtualProductException, Aggregation, Measurement, VirtualDatasetBox
from .stat_funcs import argpercentile, anynan, axisindex

class Percentile(Aggregation):
    """
    Per-band percentiles of observations through time.
    The different percentiles are stored in the output as separate bands.
    The q-th percentile of a band is named `{band}_PC_{q}`.

    :param q: list of percentiles to compute
    :param per_pixel_metadata: provenance metadata to attach to each pixel
    :arg minimum_valid_observations: if not enough observations are available,
                                     percentile will return `nodata`
    """

    def __init__(self, q,
                 minimum_valid_observations=0):

        if isinstance(q, Sequence):
            self.qs = q
        else:
            self.qs = [q]

        self.minimum_valid_observations = minimum_valid_observations
        self.reduction = {'time': 'year'}

    def compute(self, data):
        # calculate masks for pixel without enough data
        for var in data.data_vars:
            nodata = getattr(data[var], 'nodata', None)
            if nodata is not None:
                data[var] = data[var].where(data[var] > nodata)
        not_enough  = data.count(dim='time') < self.minimum_valid_observations

        def single(q):
            stat_func = partial(xarray.Dataset.reduce, dim='time', keep_attrs=True,
                                func=argpercentile, q=q)
            result = stat_func(data)

            def index_dataset(var): 
                return axisindex(data[var.name].values, var.values) 

            result = result.apply(index_dataset, keep_attrs=True)

            def mask_not_enough(var):
                nodata = getattr(data[var.name], 'nodata', -1)
                var.values[not_enough[var.name]] = nodata
                var.attrs['nodata'] = nodata
                return var

            return result.apply(mask_not_enough, keep_attrs=True).rename({var: var + '_PC_' + str(q) for var in result.data_vars})

        result = xarray.merge(single(q) for q in self.qs)
        result.attrs['crs'] = data.attrs['crs']
        return result

    def measurements(self, input_measurements):
        renamed = dict()
        for key, m in input_measurements.items():
            for q in self.qs:
                renamed[key + '_PC_' + str(q)] = Measurement(**{**m, 'name': key + '_PC_' + str(q)})
        return renamed

    def datasets(self, input_datasets):
        output_datasets = [] 
        for dim, unit in self.reduction.items():
            if dim != 'time':
                raise("Pencentile can only reduce in time at this stage")
            if unit == 'year':
                year = np.unique(pd.DatetimeIndex(input_datasets.pile.time.values).year).astype('str')
                self.time_stamp = np.array(year, dtype='datetime64[ns]')
                print(self.time_stamp)
                for year, ar in input_datasets.pile.groupby('time.year'):
                    output_datasets.append({'aggregate': ar})
                output_datasets = xarray.DataArray(np.array(output_datasets, dtype='object'), dims=['time'], coords={'time':self.time_stamp}) 
        return VirtualDatasetBox(output_datasets, input_datasets.geobox, input_datasets.product_definitions)
