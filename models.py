import tensorflow as tf
from funcs import super_loss


def get_model(learningrate=3e-5, path=None):
    """Define and compile the tf model"""
    neurons = 128
    input = tf.keras.layers.Input(4)

    X = tf.keras.layers.Dense(neurons, "relu")(input)
    X = tf.keras.layers.Dense(neurons, "relu")(X)
    X = tf.keras.layers.Dense(neurons, "relu")(X)

    output = tf.keras.layers.Dense(neurons, "selu")(X)
    output = tf.keras.layers.Dense(1, "sigmoid", name="output")(output)

    agent = tf.keras.models.Model(inputs=[input], outputs=[output])

    if path:
        agent.load_weights(path)

    agent.compile(tf.keras.optimizers.Adam(learningrate), super_loss)

    return agent

