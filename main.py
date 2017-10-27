import sys
import tensorflow as tf
from model import build_model
from audio_dataset_generator import AudioDatasetGenerator


def main():
    audio_data_path      = "assets"
    learning_rate        = 0.001
    amount_epochs        = 5
    batch_size           = 64
    dropout              = 0.2
    rnn_type             = "lstm"
    activation           = tf.nn.relu
    optimiser            = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    fft_settings         = [1024, 512, 256]
    fft_size             = fft_settings[0]
    window_size          = fft_settings[1]
    hop_size             = fft_settings[2]
    rnn_number_units     = int(fft_size / 2 + 1)
    fully_connected_dim  = int(fft_size / 2 + 1)
    sequence_length      = 16
    force_new_dataset    = False
    sample_rate          = 44100
    loss_type            = "mse"

    dataset = AudioDatasetGenerator(fft_size, window_size, hop_size,
                                    sequence_length, sample_rate)
    dataset.load(audio_data_path)

    x = tf.placeholder(tf.float32, dataset.get_x_shape())
    y = tf.placeholder(tf.float32, dataset.get_y_shape())
    training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    weights = {
        'out': tf.Variable(tf.random_normal([fully_connected_dim, fully_connected_dim]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([fully_connected_dim]))
    }

    prediction = build_model(x, activation, rnn_type, rnn_number_units,
                             weights, biases, dataset.get_y_shape()[1],
                             keep_prob)

    if loss_type == "mse":
        cost = tf.reduce_mean(tf.pow(tf.subtract(y, prediction), 2))
    elif loss_type == "l2"
        cost = tf.nn.l2_loss(y, prediction)

    train = optimiser.minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        try:
            while not dataset.completed_all_epochs(amount_epochs):

                batch_x, batch_y = dataset.get_next_batch(batch_size)

                sess.run([train], feed_dict={x: batch_x,
                                             y: batch_y,
                                             training: True,
                                             keep_prob: dropout})

                if dataset.is_new_epoch():
                    c = sess.run([cost], feed_dict={x: batch_x,
                                                    y: batch_y,
                                                    training: True,
                                                    keep_prob: dropout})

                    sys.stdout.write('Epoch: {}, Cost: {}  \r'.format(dataset.get_epoch(), c[0]))
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("Training interrupted. Generating samples.")


    dataset.generate_samples(prediction, x, training, keep_prob, 5, 300, "results")


if __name__ == "__main__":
    main()