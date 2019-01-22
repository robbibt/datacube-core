import xarray
import numpy as np

from collections import Sequence
from functools import partial
from .impl import VirtualProductException, Aggregation, Measurement
from .stat_funcs import argpercentile, anynan

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
                 minimum_valid_observations=0,
                 per_pixel_metadata=None):

        if isinstance(q, Sequence):
            self.qs = q
        else:
            self.qs = [q]

        self.minimum_valid_observations = minimum_valid_observations

    def compute(self, data):
        # calculate masks for pixel without enough data
        for var in data.data_vars:
            nodata = getattr(var, 'nodata', -1)
            data[var] = data[var].where(data[var] > nodata)
        not_enough  = data.count(dim='time') < self.minimum_valid_observations

        def single(q):
            stat_func = partial(xarray.Dataset.reduce, dim='time', keep_attrs=True,
                                func=argpercentile, q=q)

            renamed = data.rename({var: var + '_PC_' + str(q)
                                    for var in data.data_vars})
            result = stat_func(renamed)
            def mask_not_enough(var):
                nodata = getattr(var, 'nodata', -1)
                var.values[not_enough[var.name.split('_')[0]]] = nodata
                return var

            return result.apply(mask_not_enough, keep_attrs=True)

        return xarray.merge(single(q) for q in self.qs)

    def measurements(self, input_measurements):
        renamed = [Measurement(**{**m, 'name': key + '_PC_' + str(q)})
                   for q in self.qs
                   for key, m in input_measurements.items()]
        return renamed
