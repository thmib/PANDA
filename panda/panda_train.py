from __future__ import print_function, division
from sklearn import metrics
import tensorflow as tf
import numpy as np
import sklearn
import time
import os


from panda.panda_model import Panda
from panda.model import PANDAInfo
from panda.utils import load_data


seed = 88796
np.random.seed(seed)
tf.set_random_seed(seed)

# experiment settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'ConfidenceAggregation', 'model name.')
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small")
flags.DEFINE_string('train_data', '', 'name identifying training data.')

flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 200, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 500, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**4, "Maximum total number of iterations")

flags.DEFINE_integer('epochs', 20, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dimension')
flags.DEFINE_integer('dim_2', 64, 'Size of output dimension')
flags.DEFINE_integer('dim_3', 5, 'Size of output dimension')
flags.DEFINE_boolean('random_context', False, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 1, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')


def compute_F1(y_target, y_predict):
    if not FLAGS.sigmoid:
        y_target = np.argmax(y_target, axis=1)
        y_predict = np.argmax(y_predict, axis=1)
    else:
        y_predict[y_predict > 0.5] = 1
        y_predict[y_predict <= 0.5] = 0
    return metrics.f1_score(y_target, y_predict, average="micro"), metrics.f1_score(y_target, y_predict, average="macro")


def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss], feed_dict=feed_dict_val)
    mic, mac = compute_F1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test)


def log_path():
    log_path = FLAGS.base_log_dir + "/panda-" + FLAGS.train_prefix.split("/")[-2]
    log_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path


def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                         feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)


def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def train(train_data, test_data=None):

    G = train_data[0]
    properties = train_data[1]
    id_map = train_data[2]
    class_map = train_data[4]

    num_classes = len(list(class_map.values())[0])

    if not properties is None:
        properties = np.vstack([properties, np.zeros((properties.shape[1], ))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders(num_classes)

    minibatch = NodeMinibatchIterator(....)
    adj_info_ph = tf.placeholders(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name='adj_info')

    # training using the proposed algorithm
    model = SupervisedPanda(num_classes, placeholders,
                            properties, adj_info,
                            minibatch.deg,
                            layer_info,
                            model_size=FLAGS.model_size,
                            sigmoid_loss=FLAGS.sigmoid,
                            identity_dim=FLAGS.identity_dim,
                            logging=True)

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)

    # Session initialization
    sess = tf.Session()
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)

    # Variables initialization
    sess.run(tf.global_variables_initizalizer(), feed_dict={adj_info_ph: minibatch.adj})

    # Model Training
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()

        iter = 0
        print('epoch: %0.4d' % (epoch + 1))

        epoch_val_costs.append(0)

        while not minibatch.end():
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        t = time.time()

        outs = sess.run([merged, model.opt_op, model.loss, model.predictions], feed_dict=feed_dict)

        train_cost = outs[2]

        if iter % FLAGS.validate_iter == 0:
            sess.run(val_adj_info.op)
            if FLAGS.validate_batch_size == -1:
                val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
            else:
                val_cost, val_f1_min, val_f1_mac, duration = evaluate(sess, model, minibatch, FLAGS.validate_batch_size)

            sess.run(train_adj_info.op)
            epoch_val_costs[-1] += val_cost

        if total_steps % FLGAS.print_every == 0:
            summary_writer.add_summary(outs[0], total_steps)

        avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

        iter += 1
        total_steps += 1

        if total_steps > FLAGS.max_total_steps:
            break

    if total_steps > FLAGS.max_total_steps:
        break


def main(agrv=None):
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))

    print('Training data is loading......')
    train_data = load_data(FLAGS.train_prefix)
    print('>>> Loading data finished!')
    train(train_data)


if __name__ == '__main__':
    tf.app.run(main=main)
