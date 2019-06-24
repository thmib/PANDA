import tensorflow as tf
# DISCLAMER:
# Parts of this code are derived from
# https://github.com/williamleif/GraphSAGE


def masked_logit_cross_entropy(predictions, labels, mask):

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions)
    loss = tf.math.reduce_sum(loss, axis=1)
    mask = tf.dtypes.cast(mask, dtype=tf.dtypes.float32)
    mask /= tf.math.maximum(tf.math.reduce_sum(mask), tf.constant([1.]))
    loss *= mask
    return tf.math.reduce_mean(loss)


def masked_accuracy(predictions, labels, mask):
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
