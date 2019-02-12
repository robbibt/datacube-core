from typing import Mapping, Any

from .impl import VirtualProduct, Transformation, VirtualProductException
from .transformations import MakeMask, ApplyMask, ToFloat, Rename, Select
from .transformations import Mean, year, month, week, day
from .utils import reject_keys

from datacube.model import Measurement
from datacube.utils import import_function
from datacube.utils.documents import parse_yaml

__all__ = ['construct', 'Transformation', 'Measurement']


class NameResolver:
    """ Apply a mapping from name to callable objects in a recipe. """

    def __init__(self, lookup_table):
        self.lookup_table = lookup_table

    def construct(self, **recipe) -> VirtualProduct:
        """ Validate recipe and construct virtual product. """

        get = recipe.get

<<<<<<< HEAD
        kind_keys = {key for key in recipe if key in ['product', 'transform', 'aggregate', 'collate', 'juxtapose']}
        if len(kind_keys) < 1:
            raise VirtualProductException("virtual product kind not specified in {}".format(recipe))
        elif len(kind_keys) > 1:
            raise VirtualProductException("ambiguous kind in {}".format(recipe))

        if 'product' in recipe:
            def resolve_func(key, value):
                if key not in ['fuse_func', 'dataset_predicate']:
                    return value

                if callable(value):
                    return value
=======
        def lookup(name, namespace=None, kind='transformation'):
            if callable(name):
                return name
>>>>>>> d6a6c0919ff3bb9a79e0cf784b32506fc4da759f

            if namespace is not None and namespace in self.lookup_table and name in self.lookup_table[namespace]:
                result = self.lookup_table[namespace][name]
            else:
                try:
                    result = import_function(name)
                except (ImportError, AttributeError):
                    msg = "could not resolve {} {} in {}".format(kind, name, recipe)
                    raise VirtualProductException(msg)

            if not callable(result):
                raise VirtualProductException("{} not callable in {}".format(kind, recipe))

            return result

        kind_keys = {key for key in recipe if key in ['product', 'transform', 'collate', 'juxtapose', 'aggregate']}
        if len(kind_keys) < 1:
            raise VirtualProductException("virtual product kind not specified in {}".format(recipe))
        elif len(kind_keys) > 1:
            raise VirtualProductException("ambiguous kind in {}".format(recipe))

        if 'product' in recipe:
            func_keys = ['fuse_func', 'dataset_predicate']
            return VirtualProduct({key: value if key not in func_keys else lookup(value, kind='function')
                                   for key, value in recipe.items()})

        if 'transform' in recipe:
            cls_name = recipe['transform']
            input_product = get('input')

            if input_product is None:
                raise VirtualProductException("no input for transformation in {}".format(recipe))

            return VirtualProduct(dict(transform=lookup(cls_name, 'transform'),
                                       input=self.construct(**input_product),
                                       **reject_keys(recipe, ['transform', 'input'])))

        if 'aggregate' in recipe:
            def resolve_aggregate(cls_name):
                if callable(cls_name):
                    return cls_name

                if cls_name in self.lookup_table:
                    cls = self.lookup_table[cls_name]
                else:
                    try:
                        cls = import_function(cls_name)
                    except (ImportError, AttributeError):
                        msg = "could not resolve aggregation {} in {}".format(cls_name, recipe)
                        raise VirtualProductException(msg)

                if not callable(cls):
                    raise VirtualProductException("aggregation not callable in {}".format(recipe))

                return cls

            cls_name = recipe['aggregate']
            input_product = get('input')

            if input_product is None:
                raise VirtualProductException("no input for aggregation in {}".format(recipe))

            return VirtualProduct(dict(aggregate=resolve_aggregate(cls_name), input=self.construct(**input_product),
                                       **reject_keys(recipe, ['aggregate', 'input'])))

        if 'collate' in recipe:
            if len(recipe['collate']) < 1:
                raise VirtualProductException("no children for collate in {}".format(recipe))

            return VirtualProduct(dict(collate=[self.construct(**child) for child in recipe['collate']],
                                       **reject_keys(recipe, ['collate'])))

        if 'juxtapose' in recipe:
            if len(recipe['juxtapose']) < 1:
                raise VirtualProductException("no children for juxtapose in {}".format(recipe))

            return VirtualProduct(dict(juxtapose=[self.construct(**child) for child in recipe['juxtapose']],
                                       **reject_keys(recipe, ['juxtapose'])))

        if 'aggregate' in recipe:
            cls_name = recipe['aggregate']
            input_product = get('input')
            group_by = get('group_by')

            if input_product is None:
                raise VirtualProductException("no input for aggregate in {}".format(recipe))
            if group_by is None:
                raise VirtualProductException("no group_by for aggregate in {}".format(recipe))

            return VirtualProduct(dict(aggregate=lookup(cls_name, 'aggregate'),
                                       group_by=lookup(group_by, 'aggregate/group_by', kind='group_by'),
                                       input=self.construct(**input_product),
                                       **reject_keys(recipe, ['aggregate', 'input', 'group_by'])))

        raise VirtualProductException("could not understand virtual product recipe: {}".format(recipe))


# don't know if it's a good idea to keep lookup table
# it can be hundreds of lines long
DEFAULT_RESOLVER = NameResolver({'transform': dict(make_mask=MakeMask,
                                                   apply_mask=ApplyMask,
                                                   to_float=ToFloat,
                                                   rename=Rename,
                                                   select=Select),
                                 'aggregate': dict(mean=Mean),
                                 'aggregate/group_by': dict(year=year,
                                                            month=month,
                                                            week=week,
                                                            day=day)})


def construct(**recipe: Mapping[str, Any]) -> VirtualProduct:
    """
    Create a virtual product from a specification dictionary.
    """
    return DEFAULT_RESOLVER.construct(**recipe)


def construct_from_yaml(recipe: str) -> VirtualProduct:
    """
    Create a virtual product from a yaml recipe.
    """
    return construct(**parse_yaml(recipe))
