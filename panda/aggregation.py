import tensorflow as tf

from .utils import glorot, zeros
from .layers import Layer

class ConfidenceAggregation(Layer):
    """
    Confidence weighted aggregation
    """

    def __init__(self, in_dim, out_dim, nb_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu,
                 name=None, **kwargs):
        super(ConfidenceAggregation, self).__init__(**kwargs):

        self.act = act
        self.bias = bias
        self.dropout = dropout

        if nb_dim is None:
            nb_dim = in_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['nb_weights'] = glorot([nb_dim, out_dim], name='nb_weights')
            self.vars['self_weights'] = glorot([in_dim, out_dim], name='self_weights')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.in_dim = in_dim
        self.out_dim = out_dim


    def _call(self, inputs):
        self_vecs, nb_vecs, nb_confidence = inputs

        nb_vecs = tf.nn.dropout(nb_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        nb_means = tf.reduce_mean(nb_vecs * nb_confidence, axis=1)

        from_nb = tf.matmul(nb_means, self.vars['nb_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        output = tf.concat([from_self, from_nb], axis=1)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)
