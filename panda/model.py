import tensorflow as tf
import math
import .layers as layers
import .eval_metrics as metrics

from .aggregation import ConfidenceAggregation
from collections import namedtuple

flags = tf.app.flags
FLAGS = flags.FLAGS


PANDAInfo = namedtuple("PANDAInfo",
    ['layer_name', 'num_samples', 'output_dim'])


class Model(object):
    def __init__(self, **kwarg):
        valid_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in valid_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.logging = kwargs.get('logging', False)

        self.vars = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.in_data = None
        self.out_data = None
        self.loss = 0.
        self. acc = 0.
        self.optimizer = None
        self.opt_op = None


    def _build(self):
        raise NotImplementedError


    def build(self):
        with tf.variable_scope(self.name):
            self.__build()


        self.activations.append(self.in_data)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        self.outputs = self.activations[-1]

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in vars}

        self._loss()
        self._acc()
        self.opt_op = self.optimizer.minimize(self.loss)


    def _loss(self):
        raise NotImplementedError

    def _acc(self):
        raise NotImplementedError


    def save(self, sess=None):
        if not sess:
            raise AttributeError("No session is provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "temp/%s.ckpt" % self.name)


    def load(self, sess=None):
        if not sess:
            raise AttributeError("No session is provided.")
        loader = tf.train.Saver(self.vars)
        load_path = "temp/%s.ckpt" % self.name
        loader.restore(sess, load_path)
