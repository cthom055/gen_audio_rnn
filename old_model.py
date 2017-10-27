import tensorflow as tf

def build_multi_rnn(net, rnn_type, number_rnn_layers, rnn_number_units, keep_prob):
    """Builds a multilayer rnn with type of the users choosing."""
    with tf.variable_scope("multilayer_lstm"):
        cells = []
        for _ in range(number_rnn_layers):
            if rnn_type == "lstm":
                cell = tf.contrib.rnn.LSTMCell(rnn_number_units)
            elif rnn_type == "gru":
                cell = tf.contrib.rnn.GRUCell(rnn_number_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            cells += [cell]

        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        net, state = tf.nn.dynamic_rnn(cell, net, dtype=tf.float32)
        net = tf.transpose(net, [1, 0, 2])
        return tf.gather(net, int(net.get_shape()[0]) - 1)

def build_model(net, rnn_type, number_rnn_layers, rnn_number_units, weights, biases,
                output_size, keep_prob):
    """We build the TensorFlow graph here."""
    net = build_multi_rnn(net, rnn_type, number_rnn_layers, rnn_number_units, keep_prob)
    
    with tf.variable_scope("linear_layer"):  
        net = tf.add(tf.matmul(net, weights['out']), biases['out'])
    return net