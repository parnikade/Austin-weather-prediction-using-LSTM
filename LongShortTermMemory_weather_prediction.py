import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def window_data(data, window_size):
    x = []
    y = []
    i = 0
    while (i+window_size) <= len(data)-1:
        x.append(data[i:i+window_size])
        y.append(data[i+window_size])
        i += 1

    assert len(x) == len(y)
    return x, y


def LSTM_cell(input, output, state, weights_input_gate, weights_input_hidden,
              bias_input, weights_forget_gate, weights_forget_hidden,
              bias_forget, weights_output_gate, weights_output_hidden,
              bias_output, weights_memory_cell, weights_memory_cell_hidden,
              bias_memory_cell):
    input_gate = tf.sigmoid(
        tf.matmul(input, weights_input_gate) + tf.matmul(output, weights_input_hidden) + bias_input)

    forget_gate = tf.sigmoid(
        tf.matmul(input, weights_forget_gate) + tf.matmul(output,
                                                          weights_forget_hidden) + bias_forget)

    output_gate = tf.sigmoid(
        tf.matmul(input, weights_output_gate) + tf.matmul(output,
                                                          weights_output_hidden) + bias_output)

    memory_cell = tf.tanh(
        tf.matmul(input, weights_memory_cell) + tf.matmul(output,
                                                          weights_memory_cell_hidden) + bias_memory_cell)

    state = state * forget_gate + input_gate * memory_cell

    output = output_gate * tf.tanh(state)
    return state, output


def main():
    load = pd.read_csv('austin_weather.csv')
    target_data = load['TempAvgF'].values
    print(target_data)
    scalar = StandardScaler()
    scaled_data = scalar.fit_transform(target_data.reshape(-1, 1))
    x, y = window_data(scaled_data, 4)

    x_train = np.array(x[:1100])
    y_train = np.array(y[:1100])

    x_test = np.array(x[1100:])
    y_test = np.array(y[1100:])

    batch_size = 10
    window_size = 4
    hidden_layer = 256
    clip_margin = 4
    learning_rate = 0.001
    epochs = 60

    inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1])
    targets = tf.placeholder(tf.float32, [batch_size, 1])

    # weights and implementation of LSTM cell
    # LSTM weights

    # Weights for the input gate
    weights_input_gate = tf.Variable(
        tf.truncated_normal([1, hidden_layer], stddev=0.05))
    weights_input_hidden = tf.Variable(
        tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
    bias_input = tf.Variable(tf.zeros([hidden_layer]))

    # weights for the forgot gate
    weights_forget_gate = tf.Variable(
        tf.truncated_normal([1, hidden_layer], stddev=0.05))
    weights_forget_hidden = tf.Variable(
        tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
    bias_forget = tf.Variable(tf.zeros([hidden_layer]))

    # weights for the output gate
    weights_output_gate = tf.Variable(
        tf.truncated_normal([1, hidden_layer], stddev=0.05))
    weights_output_hidden = tf.Variable(
        tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
    bias_output = tf.Variable(tf.zeros([hidden_layer]))

    # weights for the memory cell
    weights_memory_cell = tf.Variable(
        tf.truncated_normal([1, hidden_layer], stddev=0.05))
    weights_memory_cell_hidden = tf.Variable(
        tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
    bias_memory_cell = tf.Variable(tf.zeros([hidden_layer]))

    # Output layer weights
    weights_output = tf.Variable(
        tf.truncated_normal([hidden_layer, 1], stddev=0.05))
    bias_output_layer = tf.Variable(tf.zeros([1]))

    outputs = []
    for i in range(batch_size):
        batch_state = np.zeros([1, hidden_layer], dtype=np.float32)
        batch_output = np.zeros([1, hidden_layer], dtype=np.float32)
        for j in range(window_size):
            batch_state, batch_output = LSTM_cell(tf.reshape(inputs[i][j], (-1, 1)),
                                                  batch_state, batch_output,
                                                  weights_input_gate,
                                                  weights_input_hidden,
                                                  bias_input,
                                                  weights_forget_gate,
                                                  weights_forget_hidden,
                                                  bias_forget,
                                                  weights_output_gate,
                                                  weights_output_hidden,
                                                  bias_output,
                                                  weights_memory_cell,
                                                  weights_memory_cell_hidden,
                                                  bias_memory_cell)
        outputs.append(tf.matmul(batch_output, weights_output) +
                       bias_output_layer)
    losses = []

    for i in range(len(outputs)):
        losses.append(tf.losses.mean_squared_error
                      (tf.reshape(targets[i], (-1, 1)), outputs[i]))

    loss = tf.reduce_mean(losses)

    # we define optimizer with gradient clipping
    gradients = tf.gradients(loss, tf.trainable_variables())
    clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    trained_optimizer = optimizer.apply_gradients(zip(gradients,
                                                      tf.trainable_variables()))
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    trained_scores = []
    for i in range(epochs):
        j = 0
        epoch_loss = []
        while (j + batch_size) <= len(x_train):
            x_batch = x_train[j:j + batch_size]
            y_batch = y_train[j:j + batch_size]

            o, c, _ = session.run([outputs, loss, trained_optimizer],
                                  feed_dict={inputs: x_batch,
                                             targets: y_batch})

            epoch_loss.append(c)
            trained_scores.append(o)
            j += batch_size
        if (i % 30) == 0:
            print('Epoch {}/{}'.format(i, epochs),
                  ' Current loss: {}'.format(np.mean(epoch_loss)))
    sup = []
    for i in range(len(trained_scores)):
        for j in range(len(trained_scores[i])):
            sup.append(trained_scores[i][j][0])

    tests = []
    i = 0
    while i + batch_size <= len(x_test):
        o = session.run([outputs],
                        feed_dict={inputs: x_test[i:i + batch_size]})
        i += batch_size
        tests.append(o)

    tests_new = []
    for i in range(len(tests)):
        for j in range(len(tests[i][0])):
            tests_new.append(tests[i][0][j])
    test_results = []
    for i in range(1311):
        if i >= 1101:
            test_results.append(tests_new[i - 1101])
        else:
            test_results.append(None)
    plt.figure(figsize=(16, 7))
    plt.title('Austin Avg temperature from 2013-12-21 to 2017-07-31')
    plt.xlabel('Index (Days)')
    plt.ylabel('Scaled average temperature in F')
    plt.plot(scaled_data, label='Original data')
    # plt.plot(sup, label='Training data')
    plt.plot(test_results, label='Testing data')
    plt.legend()
    plt.show()

main()

