import tensorflow as tf
import .layers as layers
import .model as model
from .aggregation import ConfidenceAggregation
from .utils import load_data


flags = tf.app.flags
FLAGS = flags.FLAGS


class Panda():

    def __init__(self, placeholders, properties, adj, degrees,
                 layer_infos, model_size="small", identity_dim=0, **kwargs):

        models.GeneralizedModel.__init__(self, **kwargs)
        self.aggregator_cls = ConfidenceAggregation
        self.model_size = model_size
        self.adj_info = adj
        self.in_data = placeholders["batch"]
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None

        if properties is None:
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(properties, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)

        self.layer_infos = layer_infos
        self.degrees = degrees
        self.num_classes = 2
        self.dimension = [(0 if properties is None else properties.shape[1]) + identity_dim]
        self.dimension.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()


    def predict(self):
        return tf.nn.sigmoid(self.node_preds)


    def _loss(self):            
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.node_preds,
                labels=self.placeholders['labels']))

        tf.summary.scalar('loss', self.loss)


    def build(self):
        sample, support_sizes = self.sample(self.inputs1, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

        self.output, self.aggregators = self.aggregate(samples, [self.features], self.dims, num_samples,
                support_sizes, concat=self.concat, model_size=self.model_size)

        dim_mult = 2
        self.output = tf.nn.l2_normalize(self.output, 1)
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
                         dropout=self.placeholders['dropout'], act=lambda x : x)
        self.node_preds = self.node_pred(self.outputs) # TF graph management

        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()
