# DISCLAMER:
# Parts of this code are derived from
# https://github.com/williamleif/GraphSAGE
from __future__ import print_function, division

import tensorflow as tf
from .utils import zeros

flags = tf.app.flags
FLAGS = flags.FLAGS

_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):

    def __init__(self, **kwargs):
        valid_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in valid_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')

        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))

        self.name = name
        self.vars = {}
        self.logging = kwargs.get('logging', False)
        self.sparse_in_data = False


    def _call(self, in_data):
        return in_data


    def __call__(self, in_data):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_in_data:
                tf.summary.histogram(self.name + '/in_data', in_data)

            out_data = self._call(in_data)

            if self.logging:
                tf.summary.histogram(self.name + '/out_data', out_data)

            return out_data


    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])
